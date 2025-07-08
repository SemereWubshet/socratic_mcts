import argparse
import gc
import json
import math
import pathlib
import shutil
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import scipy
import torch
import unsloth
from datasets import load_dataset, Dataset
from ollama import Client
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForSequenceClassification
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

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


class ActionValueFn:

    def __init__(self, base_model: str, max_length: int = 1024, gpu: Optional[str] = None):
        self._max_length = max_length
        self._base_model = base_model
        self.device = torch.device(gpu) if gpu is not None else None
        self.model = None
        self.tokenizer = None

    def __call__(self, history: List[Dict[str, str]]) -> float:
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

        if self.device is not None:
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            value = float(self.model(**inputs).logits)

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
        self.model = AutoModelForSequenceClassification.from_pretrained(self._base_model, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.chat_template = (
            "{% for message in messages if not message['role'] == 'system' %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
        )
        if self.device is not None:
            self.model = self.model.to(self.device)

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

    def load(self) -> None:
        model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self._base_model,
            dtype=torch.bfloat16,
            max_seq_length=self.max_length,
            load_in_4bit=False,  # False for LoRA 16bit
            load_in_8bit=False,
        )
        self.model = unsloth.FastLanguageModel.for_inference(model)

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
    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(ollama_client))
    seed_dataset = SeedDataset.model_validate_json(pathlib.Path(dataset_path).read_text())

    model = Qwen(base_model=str(policy_path))

    interactions_dataset = gen_teacher_student_interactions(
        seed_dataset, nemo, SimpleTeacher(model), max_interactions=max_interactions
    )
    output_path.write_text(interactions_dataset.model_dump_json(indent=4))

    model.unload()


def vf_rollout(dataset_path: pathlib.Path, action_vf_path: str, output_path: pathlib.Path, device: str) -> None:
    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    action_value_fn = ActionValueFn(action_vf_path, max_length=1024, gpu=device)

    dataset = defaultdict(list)
    for item in evaluations_dataset.evaluations:
        assessment = item.assessment
        history = item.interaction.chat_history.root
        trajectory = [
            [
                {"role": "user" if h.role == "Student" else "assistant", "content": h.content} for h in history[:2 * z]
            ] for z in range(1, math.ceil(len(history) / 2))
        ]
        values = [action_value_fn(h) for h in trajectory]

        gamma = 1.
        _lambda = 0.9
        vf_preds = np.array(values)
        rwd = np.zeros(vf_preds.shape[0])
        rwd[-1] = np.float32(1. if assessment else -1.)
        vpred_t = np.concatenate((vf_preds, [0.]))
        delta_t = -vpred_t[:-1] + rwd + gamma * vpred_t[1:]
        advantages = discount_cumsum(delta_t, gamma * _lambda)
        value_targets = advantages + vf_preds

        dataset["history"].extend(trajectory)
        dataset["labels"].extend(value_targets)

    hf_dataset = Dataset.from_dict(dataset)
    tokenized_dataset = hf_dataset.map(action_value_fn.batch_tokenize, batched=True, batch_size=8).shuffle()
    tokenized_dataset.save_to_disk(output_path)

    action_value_fn.unload()


def vf_train(
        dataset_path: pathlib.Path,
        value_checkpoints_path: pathlib.Path,
        action_value_fn_path: str,
        vf_output_path: pathlib.Path,
        device: str
) -> None:
    tokenized_dataset = Dataset.load_from_disk(dataset_path)
    training_args = TrainingArguments(output_dir=value_checkpoints_path, num_train_epochs=1, learning_rate=1e-5,
                                      gradient_checkpointing=True)
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=1024, gpu=device)
    action_value_fn.load()
    trainer = Trainer(model=action_value_fn.model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    action_value_fn.save(vf_output_path)

    action_value_fn.unload()


def prepare_for_dpo(
        evaluations_dataset_path: pathlib.Path,
        action_value_fn_path: str,
        policy_path: pathlib.Path,
        output_path: pathlib.Path,
        device: str
) -> None:
    model = Qwen(base_model=str(policy_path))
    evaluations_dataset = EvaluationDataset.model_validate_json(evaluations_dataset_path.read_text())
    completions = []
    for item in evaluations_dataset.evaluations:
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
                    {"role": "assistant", "content": c1},
                    {"role": "assistant", "content": c2}
                )
            )
    model.unload()

    action_value_fn = ActionValueFn(action_value_fn_path, max_length=1024, gpu=device)

    dataset = {"prompt": [], "chosen": [], "rejected": [], "v_chosen": [], "v_rejected": []}
    for t, c1, c2 in completions:
        v1 = action_value_fn(t + [c1, ])
        v2 = action_value_fn(t + [c2, ])

        dataset["prompt"].append(t)

        if v1 >= v2:
            dataset["chosen"].append(c1)
            dataset["v_chosen"].append(v1)
            dataset["rejected"].append(c2)
            dataset["v_rejected"].append(v2)
        else:
            dataset["chosen"].append(c2)
            dataset["v_chosen"].append(v2)
            dataset["rejected"].append(c1)
            dataset["v_rejected"].append(v1)

    action_value_fn.unload()

    with open(output_path, "w", encoding="UTF-8") as f:
        f.write(json.dumps(dataset))


def policy_train(
        dataset_path: pathlib.Path,
        policy_path: pathlib.Path,
        checkpoints_dir: pathlib.Path,
        output_dir: pathlib.Path
) -> None:
    train_dataset = json.loads(dataset_path.read_text(encoding="UTF-8"))

    qwen = Qwen(str(policy_path))
    qwen.load()

    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=1,
        max_length=1024,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir=checkpoints_dir,
        beta=0.1,
        learning_rate=1e-6,
        max_prompt_length=128,
    )

    dpo_trainer = DPOTrainer(
        model=qwen.model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=qwen.tokenizer
    )

    dpo_trainer.train()
    qwen.save(output_dir)
    qwen.unload()


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
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            for c in examples["messages"]
        ]
        return {"text": _input}

    dataset = dataset.map(prepare_prompts, batched=True)

    training_args = SFTConfig(
        max_seq_length=1024,
        output_dir=train_dir / "stf",
    )
    trainer = SFTTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    train_stats = trainer.train()
    with open(train_dir / "train_stats.json", "w") as f:
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
    parser.add_argument(
        "--num-conversations", required=True, type=int, help="Number of training examples on each iteration"
    )
    parser.add_argument(
        "--max-interactions", default=8, type=int, help="Max number of student-teacher rounds"
    )
    parser.add_argument(
        "--ollama-client", type=str, default="http://atlas1api.eurecom.fr", help="Address to ollama server"
    )
    parser.add_argument(
        "--vf-training-it", type=int, default=6, help="Number of action-value function training steps"
    )
    parser.add_argument(
        "--base-model", type=str, default="phi4", choices=["phi4", "smollm"], help="Base model for PEFT"
    )
    parser.add_argument("--train-batch-size", type=int, default=1, help="Batch size for training")

    parser.add_argument("--clean", action="store_true", help="Clean root_dir if it exists")
    args = parser.parse_args()

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

    if not stf_pretrained.exists():
        print(f" -------------------- ------------------ starting STF -------------------- ------------------")
        stf_warmup(args.stf_dataset, train_dir, stf_pretrained)

    textbooks = load_dataset("princeton-nlp/TextbookChapters")
    for i in range(args.num_iterations):
        train_it_dir = train_dir / f"iteration_{i}"
        train_it_dir.mkdir(exist_ok=True)

        seeds_path = train_it_dir / "seeds.json"
        interactions_path = train_it_dir / "interactions.json"
        evaluations_path = train_it_dir / "evaluations.json"
        action_vfn_model_dir = train_it_dir / "action_value_fn"
        value_checkpoints = train_it_dir / "value_checkpoints"
        dpo_dataset = train_it_dir / "dpo_dataset.json"
        policy_model_dir = train_it_dir / "policy_fn"
        policy_checkpoints = train_it_dir / "policy_checkpoints"

        previous_iteration = train_dir / f"iteration_{i - 1}"
        current_policy_path = previous_iteration / "policy_fn" if i > 0 else stf_pretrained
        current_vf_path = previous_iteration / "action_value_fn" if i > 0 else "answerdotai/ModernBERT-large"

        if policy_model_dir.exists():
            continue

        print(f" -------------------- ------------------ starting it {i} -------------------- ------------------")

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

            print(f"\nAvg. perf. {i}: {evaluations_dataset.avg_performance()}\n", flush=True)

        if not action_vfn_model_dir.exists():
            print()
            print("#### VF training")
            for j in range(args.vf_training_it):
                vf_training_path = train_it_dir / "vf_training"
                current_vf_step_path = current_vf_path if j == 0 else vf_training_path / f"it_{j - 1}" / "value_fn"

                vf_target_path = (
                    vf_training_path / f"it_{j}" / "value_fn"
                    if j < args.vf_training_it - 1 else action_vfn_model_dir
                )

                dataset_path = vf_training_path / f"it_{j}" / "dataset"
                vf_rollout(evaluations_path, str(current_vf_step_path), dataset_path, "cuda")

                vf_train(
                    dataset_path,
                    value_checkpoints / f"it_{j}",
                    str(current_vf_step_path),
                    vf_target_path,
                    device="cuda"
                )

        print()
        print("#### Preparing for DPO training")
        if not dpo_dataset.exists():
            prepare_for_dpo(
                evaluations_path,
                action_vfn_model_dir,
                current_policy_path,
                dpo_dataset,
                device="cuda"
            )

        print()
        print("#### Policy training")
        policy_train(
            dpo_dataset,
            current_policy_path,
            policy_checkpoints,
            policy_model_dir
        )
