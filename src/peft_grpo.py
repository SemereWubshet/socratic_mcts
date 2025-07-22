import argparse
import gc
import json
import math
import pathlib
import shutil
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
import scipy
import torch
import unsloth
from datasets import load_dataset, Dataset
from ollama import Client
from torch import nn
from torch.nn import MSELoss
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    ModernBertConfig, \
    ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer

from agents import Teacher, OllamaAgent, LLM
from evaluate import gen_teacher_student_interactions, gen_seeds, evaluate
from schemas import ChatHistory, SeedDataset, EvaluationDataset, InteractionDataset


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.

     .. testcode::
        :skipif: True

        x = np.array([0.0, 1.0, 2.0, 3.0])
        gamma = 0.9
        discount_cumsum(x, gamma)

    .. testoutput::

        array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
               1.0 + 0.9*2.0 + 0.9^2*3.0,
               2.0 + 0.9*3.0,
               3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


class ActionValueFunctionModel(ModernBertPreTrainedModel):
    """Heavily inspired from ModernBertForSequenceClassification, but with a Tahn value head on the top of the model."""
    _supports_flash_attn_2 = False

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.value = nn.Tanh()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if isinstance(module, ActionValueFunctionModel):
            init_weight(module.classifier, self.config.hidden_size ** -0.5)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            sliding_window_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            indices: Optional[torch.Tensor] = None,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,
            batch_size: Optional[int] = None,
            seq_len: Optional[int] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.config.classifier_pooling == "cls":
            last_hidden_state = last_hidden_state[:, 0]
        elif self.config.classifier_pooling == "mean":
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        pooled_output = self.classifier(pooled_output)
        value = self.value(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels != 1:
                    raise ValueError(f"Number of output labels must be 1, but found {self.num_labels}")
                self.config.problem_type = "regression"

            if self.config.problem_type != "regression":
                raise ValueError(f"This is a regression model, but found {self.config.problem_type}")
            loss_fct = MSELoss()
            loss = loss_fct(value.squeeze(), labels.squeeze())

        if not return_dict:
            output = (value,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=value,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ActionValueFn:

    def __init__(self, base_model: str, max_length: int = 1024):
        self._max_length = max_length
        self._base_model = base_model
        self.device = torch.device("cuda")
        self.model = None
        self.tokenizer = None

    def __call__(self, history: List[Dict[str, str]]):
        if getattr(self, "model", None) is None or self.tokenizer is None:
            self.load()

        raw_prompt = self.tokenizer.apply_chat_template(history, tokenize=False)
        tokenized = self.tokenizer(
            raw_prompt, return_tensors="pt", truncation=True, max_length=self._max_length
        )
        inputs = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask
        }

        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            value = self.model(**inputs).logits

        return value

    def batch_tokenize(self, dataset: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        raw_texts = self.tokenizer.apply_chat_template(dataset["history"], tokenize=False)
        return self.tokenizer(
            raw_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self._max_length
        )

    def save(self, path: pathlib.Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self) -> None:
        config = ModernBertConfig.from_pretrained(
            self._base_model,
            num_labels=1,
            classifier_dropout=0.05,
            torch_dtype="float32",
            problem_type="regression"
        )
        self.model = ActionValueFunctionModel.from_pretrained(
            pretrained_model_name_or_path=self._base_model,
            config=config,
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        if not pathlib.Path(self._base_model).exists() or not pathlib.Path(self._base_model).is_dir():
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_tokens(["[USER]", "[/USER]", "[EOT]"])
            self.tokenizer.chat_template = (
                "{% for i in range(0, messages|length, 2) %}"
                "{% if i + 1 < messages|length %}"
                "[USER]{{ messages[i].content }}[/USER] {{ messages[i+1].content }}[EOT]\n"
                "{% endif %}"
                "{% endfor %}"
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        # for name, param in self.model.named_parameters():
        #     if name.startswith("classifier."):
        #         print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item():.4f}")

    def unload(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


class Qwen(LLM):

    def __init__(self, base_model: str, max_length: int = 1024):
        self._base_model = base_model
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.15) -> str:
        if getattr(self, "model", None) is None or self.tokenizer is None:
            self.load()

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([raw_prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, max_new_tokens=128, do_sample=True, temperature=temperature
        )
        generation = outputs[0, len(inputs['input_ids'][0]):]
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded

    def load(self, for_inference: bool = True) -> None:
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self._base_model,
            dtype=torch.bfloat16,
            max_seq_length=self.max_length,
            load_in_4bit=False,  # False for LoRA 16bit
            load_in_8bit=False
        )

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if for_inference:
            self.model = unsloth.FastLanguageModel.for_inference(self.model)

        # ✅ Patch apply_chat_template to default enable_thinking=False
        if hasattr(self.tokenizer, "apply_chat_template"):
            original_fn = self.tokenizer.apply_chat_template

            def patched_apply_chat_template(conversation, **kwargs):
                print("in chat template: ")
                print(conversation)
                kwargs.setdefault("enable_thinking", False)
                return original_fn(conversation, **kwargs)

            self.tokenizer.apply_chat_template = patched_apply_chat_template

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return f"Qwen3 ({self._base_model})"

    def save(self, path: pathlib.Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def unload(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SimpleTeacher(Teacher):

    def chat(self, chat_history: ChatHistory) -> str:
        messages = [
            {"role": "user" if h.role == "Student" else "assistant", "content": h.content} for h in chat_history.root
        ]
        return self._llm.query(messages)

    def model_name(self) -> str:
        return self._llm.model_name


def rollout(
        dataset_path: pathlib.Path,
        policy_path: pathlib.Path,
        output_path: pathlib.Path,
        ollama_client: str,
        max_interactions: int
) -> None:
    student_llm = OllamaAgent(model="mistral-small3.1:24b", client=Client(ollama_client))
    seed_dataset = SeedDataset.model_validate_json(pathlib.Path(dataset_path).read_text())

    model = Qwen(base_model=str(policy_path))

    interactions_dataset = gen_teacher_student_interactions(
        seed_dataset, student_llm, SimpleTeacher(model), max_interactions=max_interactions
    )
    output_path.write_text(interactions_dataset.model_dump_json(indent=4))

    model.unload()
    student_llm.unload()


def vf_rollout(
        dataset_path: pathlib.Path,
        action_vf_path: str,
        output_path: pathlib.Path,
) -> Dict[str, Any]:
    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    action_value_fn = ActionValueFn(action_vf_path, max_length=1024)

    all_preds = []
    all_targets = []
    dataset = defaultdict(list)
    for item in tqdm(evaluations_dataset.get_valid(), desc="vf rollout"):
        assessment = item.assessment
        history = item.interaction.chat_history.root
        trajectory = [
            [
                {"role": "user" if h.role == "Student" else "assistant", "content": h.content} for h in history[:2 * z]
            ] for z in range(1, math.ceil(len(history) / 2))
        ]
        values = [float(action_value_fn(h)) for h in trajectory]

        gamma = 1.
        _lambda = 0.7
        vf_preds = np.array(values, dtype=np.float32)
        rwd = np.zeros(vf_preds.shape[0], dtype=np.float32)
        rwd[-1] = np.float32(1. if assessment else -1.)
        vpred_t = np.concatenate((vf_preds, [0.]))
        delta_t = -vpred_t[:-1] + rwd + gamma * vpred_t[1:]
        advantages = discount_cumsum(delta_t, gamma * _lambda)
        value_targets = advantages + vf_preds

        dataset["history"].extend(trajectory)
        dataset["labels"].extend(value_targets.astype(np.float32))

        all_preds.extend(vf_preds)
        all_targets.extend(value_targets)

    hf_dataset = Dataset.from_dict(dataset)
    tokenized_dataset = hf_dataset.map(action_value_fn.batch_tokenize, batched=True, batch_size=8)
    tokenized_dataset.save_to_disk(output_path)

    action_value_fn.unload()

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    vf_loss = float(np.mean((all_preds - all_targets) ** 2))
    explained_var = 1. - np.var(all_targets - all_preds) / (np.var(all_targets) + 1e-8)
    _min = float(np.min(all_preds))
    _max = float(np.max(all_preds))

    return {"vf_loss": vf_loss, "explained_var": explained_var, "min": _min, "max": _max}


def vf_train(
        dataset_path: pathlib.Path,
        action_value_fn_path: str,
        vf_output_path: pathlib.Path,
        value_checkpoints_path: pathlib.Path,
) -> Dict[str, Any]:
    tokenized_dataset = Dataset.load_from_disk(dataset_path)
    tokenized_dataset = tokenized_dataset.shuffle()
    training_args = TrainingArguments(
        num_train_epochs=1,
        output_dir=value_checkpoints_path,
        overwrite_output_dir=True,
        learning_rate=5e-6,
        gradient_checkpointing=True
    )
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=1024)
    action_value_fn.load()
    trainer = Trainer(model=action_value_fn.model, args=training_args, train_dataset=tokenized_dataset)
    results = trainer.train()
    action_value_fn.save(vf_output_path)

    action_value_fn.unload()

    return results.metrics


def prepare_for_dpo(
        evaluations_dataset_path: pathlib.Path,
        action_value_fn_path: str,
        policy_path: pathlib.Path,
        output_path: pathlib.Path,
) -> Dict[str, Any]:
    model = Qwen(base_model=str(policy_path))
    evaluations_dataset = EvaluationDataset.model_validate_json(evaluations_dataset_path.read_text())
    completions: List[Tuple[List[Dict[str, str]], ...]] = []
    for item in tqdm(evaluations_dataset.get_valid(), desc="DPO policy forward pass"):
        history = item.interaction.chat_history.root
        for z in range(1, math.ceil(len(history) / 2)):
            trajectory = [
                {
                    "role": "user" if h.role == "Student" else "assistant",
                    "content": h.content
                }
                for h in history[:2 * z - 1]
            ]
            c1 = model.query(trajectory, temperature=2.1)
            c2 = model.query(trajectory, temperature=2.1)
            completions.append(
                (
                    trajectory,
                    [{"role": "assistant", "content": c1}],
                    [{"role": "assistant", "content": c2}]
                )
            )
    model.unload()

    action_value_fn = ActionValueFn(action_value_fn_path, max_length=1024)

    dataset = {"prompt": [], "chosen": [], "rejected": [], "v_chosen": [], "v_rejected": []}
    for t, cl1, cl2 in tqdm(completions, desc="DPO eval samples"):
        v1 = float(action_value_fn(t + cl1))
        v2 = float(action_value_fn(t + cl2))

        dataset["prompt"].append(t)

        if v1 >= v2:
            dataset["chosen"].append(cl1)
            dataset["v_chosen"].append(v1)
            dataset["rejected"].append(cl2)
            dataset["v_rejected"].append(v2)
        else:
            dataset["chosen"].append(cl2)
            dataset["v_chosen"].append(v2)
            dataset["rejected"].append(cl1)
            dataset["v_rejected"].append(v1)

    action_value_fn.unload()

    with open(output_path, "w", encoding="UTF-8") as f:
        f.write(json.dumps(dataset))

    return {
        "mean_v_chosen": np.mean(dataset["v_chosen"]),
        "max_v_chosen": np.max(dataset["v_chosen"]),
        "min_v_chosen": np.min(dataset["v_chosen"]),
        "mean_v_rejected": np.mean(dataset["v_rejected"]),
        "max_v_rejected": np.max(dataset["v_rejected"]),
        "min_v_rejected": np.min(dataset["v_rejected"]),
    }


def policy_train(
        dataset_path: pathlib.Path,
        policy_path: pathlib.Path,
        action_value_fn_path: str,
        checkpoints_dir: pathlib.Path,
        output_dir: pathlib.Path
) -> Dict[str, Any]:
    vf = ActionValueFn(action_value_fn_path, max_length=1024)
    vf.load()

    model = Qwen(str(policy_path), max_length=1024)
    model.load(for_inference=False)

    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    dataset = defaultdict(list)
    for item in tqdm(evaluations_dataset.get_valid(), desc="DPO policy forward pass"):
        history = item.interaction.chat_history.root
        for z in range(1, math.ceil(len(history) / 2)):
            trajectory = [
                {
                    "role": "user" if h.role == "Student" else "assistant",
                    "content": h.content
                }
                for h in history[:2 * z - 1]
            ]
            dataset["prompt"].append(trajectory)

    hf_dataset = Dataset.from_dict(dataset)

    print("prompt:")
    print(hf_dataset["prompt"][0])

    # GRPO config
    training_args = GRPOConfig(
        learning_rate=1e-6,
        top_p=0.95,
        output_dir=checkpoints_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        temperature=2.4,
        max_completion_length=128,
        num_generations=8,
        num_train_epochs=1,
        save_strategy="epoch",
        report_to="none",
        generation_kwargs={
            "max_length": 1024,
        }
    )

    def rwd_fn(prompts: List[List[Dict[str, str]]], completions: List[str], **kwargs) -> List[float]:
        print("history:")
        print(prompts[0])
        print("completions: ")
        print(completions[0])
        combined = [p + [{"role": "assistant", "content": c}, ] for p, c in zip(prompts, completions)]
        print("combined: ")
        print(combined[0])
        return [float(vf(c)) for c in combined]

    old_method = model.model.generate

    def patch(*args, **kwargs):
        print(args)
        print(kwargs)
        input_ids = args[0] if len(args) > 0 else kwargs["input_ids"]
        decoded = model.tokenizer.decode(token_ids=input_ids[0])
        print(decoded)
        return old_method(*args, **kwargs)

    model.model.generate = patch

    answer = model.query([{"role":"user", "content": "what am I testing?"}])
    print(answer)
    print("now GRPO")

    trainer = GRPOTrainer(
        args=training_args,
        model=model.model,
        processing_class=model.tokenizer,
        reward_funcs=rwd_fn,
        train_dataset=hf_dataset,
    )

    results = trainer.train()

    # Save only the LoRA adapter
    model.save(output_dir)

    model.unload()
    vf.unload()

    return results.metrics


def stf_warmup(dataset_path: pathlib.Path, train_dir: pathlib.Path, pretrained_dir: pathlib.Path) -> None:
    # https://docs.unsloth.ai/get-started/fine-tuning-guide#id-2.-choose-the-right-model--method
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",
        max_seq_length=1024,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # False for LoRA 16bit
        load_in_8bit=False,
        # full_finetuning=True,  # (see https://github.com/unslothai/unsloth/issues/2713)
        gpu_memory_utilization=0.7,  # Reduce if out of memory
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128 * 2,
        use_gradient_checkpointing="unsloth",
    )

    model.gradient_checkpointing_enable()  # https://github.com/huggingface/transformers/issues/30544
    tokenizer = unsloth.get_chat_template(tokenizer, chat_template="qwen3")

    dataset = Dataset.load_from_disk(dataset_path)

    def prepare_prompts(examples) -> None:
        _input = [
            tokenizer.apply_chat_template(
                c,
                padding="max_length",
                max_length=1024,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            for c in examples["messages"]
        ]
        return {"text": _input}

    dataset = dataset.map(prepare_prompts, batched=True)

    training_args = SFTConfig(
        max_seq_length=1024,
        output_dir=train_dir / "stf",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        lr_scheduler_type="linear",
    )
    trainer = SFTTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    train_stats = trainer.train()
    with open(train_dir / "stf_train_stats.json", "w") as f:
        json.dump(train_stats, f)

    model.save_pretrained(pretrained_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FULL RL", description="Train a socratic llm through RL."
    )

    parser.add_argument("--root-dir", required=True, type=pathlib.Path, help="Path where to store pipeline artifacts")
    parser.add_argument("--num-iterations", required=True, type=int, help="Number of training iterations")
    parser.add_argument("--stf-dataset", required=True, type=pathlib.Path, help="Path to dataset to be used for STF")
    parser.add_argument("--ollama-client", type=str, required=True, help="Address to ollama server")
    parser.add_argument(
        "--num-conversations", required=True, type=int, help="Number of training examples on each iteration"
    )
    parser.add_argument(
        "--max-interactions", default=8, type=int, help="Max number of student-teacher rounds"
    )
    parser.add_argument(
        "--vf-training-it", type=int, default=3, help="Number of action-value function training steps"
    )

    parser.add_argument("--clean", action="store_true", help="Clean root_dir if it exists")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    root_dir = pathlib.Path(args.root_dir)
    root_dir.mkdir(exist_ok=True)
    if args.clean:
        for child in root_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    train_dir: pathlib.Path = root_dir / "train"
    train_dir.mkdir(exist_ok=True, parents=True)

    student = OllamaAgent(model="mistral-small3.1:24b", client=Client(args.ollama_client))

    judge = OllamaAgent(model="qwen3:32b", client=Client(args.ollama_client))

    stf_pretrained = train_dir / "stf" / "pretrained"
    policy_checkpoints = train_dir / "policy_checkpoints"
    value_checkpoints = train_dir / "value_checkpoints"
    init_q_path = train_dir / "init_q"

    if not stf_pretrained.exists():
        print(f" -------------------- ------------------ starting STF -------------------- ------------------")
        print(f"std_dataset={args.stf_dataset}")
        print(f"train_dir={train_dir}")
        print(f"stf_pretrained={stf_pretrained}")
        stf_warmup(args.stf_dataset, train_dir, stf_pretrained)

    if not init_q_path.exists():
        action_value_fn = ActionValueFn("answerdotai/ModernBERT-large", max_length=1024)
        action_value_fn.load()
        action_value_fn.save(init_q_path)
        action_value_fn.unload()
        del action_value_fn

    textbooks = load_dataset("princeton-nlp/TextbookChapters")
    for i in range(args.num_iterations):
        train_it_dir = train_dir / f"iteration_{i}"
        train_it_dir.mkdir(exist_ok=True)

        stats_path = train_it_dir / "stats.json"
        seeds_path = train_it_dir / "seeds.json"
        interactions_path = train_it_dir / "interactions.json"
        evaluations_path = train_it_dir / "evaluations.json"
        action_vfn_model_dir = train_it_dir / "action_value_fn"
        policy_model_dir = train_it_dir / "policy_fn"

        previous_iteration = train_dir / f"iteration_{i - 1}"
        current_policy_path = previous_iteration / "policy_fn" if i > 0 else stf_pretrained
        current_vf_path = previous_iteration / "action_value_fn" if i > 0 else init_q_path

        if policy_model_dir.exists():
            continue

        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
        else:
            stats: Dict[str, Any] = {"it": i}

        print(f" -------------------- ------------------ starting it {i} -------------------- ------------------")

        print(f"train_it_dir={train_it_dir}")
        print(f"stats_path={stats_path}")
        print(f"seeds_path={seeds_path}")
        print(f"interactions_path={interactions_path}")
        print(f"evaluations_path={evaluations_path}")
        print(f"action_vfn_model_dir={action_vfn_model_dir}")
        print(f"policy_model_dir={policy_model_dir}")
        print(f"previous_iteration={previous_iteration}")
        print(f"current_policy_path={current_policy_path}")
        print(f"current_vf_path={current_vf_path}")

        if not seeds_path.exists():
            seed_dataset = gen_seeds(textbooks, student, num_of_conversations=args.num_conversations)
            seeds_path.write_text(seed_dataset.model_dump_json(indent=4))

        if not interactions_path.exists():
            print()
            print("#### Rolling out policy")
            rollout(
                seeds_path,
                current_policy_path,
                interactions_path,
                args.ollama_client,
                args.max_interactions
            )

        if not evaluations_path.exists():
            print()
            print("#### Policy evaluation")
            interactions_dataset = InteractionDataset.model_validate_json(interactions_path.read_text())
            evaluations_dataset = evaluate(interactions_dataset, judge, max_interactions=args.max_interactions)
            evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))
            judge.unload()

            print()
            print(f"Avg. perf. {i}: {evaluations_dataset.avg_performance()}")
            print()
            stats["avg_perf"] = evaluations_dataset.avg_performance()
            stats_path.write_text(json.dumps(stats))

        if not action_vfn_model_dir.exists():
            print()
            print("#### VF training")
            stats["vf_training"] = {"vf_loss": [], "explained_var": [], "train": []}
            for j in range(args.vf_training_it):
                vf_training_path = train_it_dir / "vf_training"
                current_vf_step_path = current_vf_path if j == 0 else vf_training_path / f"it_{j - 1}" / "value_fn"

                vf_target_path = vf_training_path / f"it_{j}" / "value_fn"

                dataset_path = vf_training_path / f"it_{j}" / "dataset"
                d = vf_rollout(evaluations_path, str(current_vf_step_path), dataset_path)

                stats["vf_training"]["vf_loss"].append(d["vf_loss"])
                stats["vf_training"]["explained_var"].append(d["explained_var"])

                print(f"evaluations_path={evaluations_path}")
                print(f"dataset_path={dataset_path}")
                print(f"current_vf_step_path={current_vf_step_path}")
                print(f"vf_target_path={vf_target_path}")
                print(f"dataset_path={value_checkpoints}")

                d = vf_train(
                    dataset_path,
                    str(current_vf_step_path),
                    vf_target_path,
                    value_checkpoints
                )

                if j >= args.vf_training_it - 1:
                    action_vfn_model_dir.symlink_to(vf_target_path, target_is_directory=True)

                stats["vf_training"]["train"].append(d)
                stats_path.write_text(json.dumps(stats))

        print()
        print("#### Policy training")

        print(f"evaluations_oath={evaluations_path}")
        print(f"current_policy_path={current_policy_path}")
        print(f"policy_checkpoints={policy_checkpoints}")
        print(f"action_vfn_model_dir={action_vfn_model_dir}")
        print(f"policy_model_dir={policy_model_dir}")

        stats["grpo"] = policy_train(
            evaluations_path,
            current_policy_path,
            action_vfn_model_dir,
            policy_checkpoints,
            policy_model_dir
        )

        print(stats)

        stats_path.write_text(json.dumps(stats))
