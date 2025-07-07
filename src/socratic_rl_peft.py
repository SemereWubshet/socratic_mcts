import argparse
import gc
import json
import math
import pathlib
import shutil
from collections import defaultdict
from multiprocessing import Process
from typing import List, Dict, Optional

import numpy as np
import scipy
import torch
import unsloth
from datasets import load_dataset, Dataset
from ollama import Client
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from agents import Teacher, OllamaAgent, LLM
from evaluate import gen_teacher_student_interactions, gen_seeds
from schemas import ChatHistory, SeedDataset, EvaluationDataset


# https://huggingface.co/unsloth/Qwen3-4B-Base
# https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune#fine-tuning-qwen3-with-unsloth
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False  # Disables thinking mode
# )

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
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = (
            "{% for message in messages if not message['role'] == 'system' %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)

        if self.device is not None:
            self.model = self.model.to(self.device)

    def __call__(self, history: List[Dict[str, str]]) -> float:
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


class Qwen(LLM):

    def __init__(self, base_model: str, max_length: int = 1024):
        self._base_model = base_model
        model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=base_model,
            dtype=torch.bfloat16,
            max_seq_length=max_length,
            load_in_4bit=False,  # False for LoRA 16bit
            load_in_8bit=False,
        )
        self.model = unsloth.FastLanguageModel.for_inference(model)

        self.max_length = max_length

    def query(self, messages: List[Dict[str, str]]) -> str:
        raw_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([raw_prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, max_new_tokens=128, do_sample=True, temperature=0.15
        )
        generation = outputs[0, len(inputs['input_ids'][0]):]
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return f"Qwen3 ({self._base_model})"

    def save(self, path: pathlib.Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def unload(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SimpleTeacher(Teacher):

    def chat(self, chat_history: ChatHistory) -> str:
        h = chat_history.format()
        return self._llm.query(h)

    def model_name(self) -> str:
        return self._llm.model_name


def rollout(
        dataset_path: pathlib.Path,
        policy_path: Optional[pathlib.Path],
        output_path: pathlib.Path,
        ollama_client: str,
        max_interactions: int
) -> None:
    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(ollama_client))
    seed_dataset = SeedDataset.model_validate_json(pathlib.Path(dataset_path).read_text())

    model = Qwen(base_model=policy_path)

    interactions_dataset = gen_teacher_student_interactions(
        seed_dataset, nemo, SimpleTeacher(model), max_interactions=max_interactions
    )
    output_path.write_text(interactions_dataset.model_dump_json(indent=4))


def vf_rollout(dataset_path: pathlib.Path, action_vf_path: str, output_path: pathlib.Path, device: str) -> None:
    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    action_value_fn = ActionValueFn(action_vf_path, max_length=768, gpu=device)

    dataset = defaultdict(list)
    for item in evaluations_dataset.root:
        assessment = item.assessment
        history = item.interaction.chat_history.root
        trajectory = [ChatHistory.model_validate(history[:2 * z]).format() for z in
                      range(1, math.ceil(len(history) / 2))]
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


def vf_train(
        dataset_path: pathlib.Path,
        value_checkpoints_path: pathlib.Path,
        action_value_fn_path: str,
        vf_output_path: pathlib.Path
) -> None:
    tokenized_dataset = Dataset.load_from_disk(dataset_path)
    training_args = TrainingArguments(output_dir=value_checkpoints_path, num_train_epochs=1, learning_rate=1e-5,
                                      gradient_checkpointing=True)
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=768)
    trainer = Trainer(model=action_value_fn.model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    action_value_fn.save(vf_output_path)


def policy_train(
        dataset_path: pathlib.Path,
        policy_checkpoints: pathlib.Path,
        action_value_fn_path: str,
        policy_path: Optional[pathlib.Path],
        output_dir: pathlib.Path,
        base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
) -> None:
    # Load dataset
    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    dataset = defaultdict(list)
    for item in evaluations_dataset.root:
        history = item.interaction.chat_history.root
        trajectory = [ChatHistory.model_validate(history[:2 * z + 1]).format() for z in
                      range(math.ceil(len(history) / 2))]
        dataset["prompt"].extend(trajectory)
    hf_dataset = Dataset.from_dict(dataset)

    # Define reward model
    rwd_fn = ActionValueFn(action_value_fn_path, max_length=768)

    # GRPO config
    training_args = GRPOConfig(
        output_dir=policy_checkpoints,
        learning_rate=5e-6,  # higher LR for faster adaptation if LoRA is used
        per_device_train_batch_size=args.train_batch_size,
        temperature=1.2,  # lower to reduce randomness
        max_prompt_length=880,  # allow richer context
        max_completion_length=128,  # generate more thoughtful responses
        num_generations=4,  # increase diversity
        num_train_epochs=1,  # train longe
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        save_strategy="epoch",
        logging_steps=50,
        evaluation_strategy="no",  # consider "epoch" if val set is added
        warmup_steps=100,
        lr_scheduler_type="linear",
        report_to="none",  # add "wandb" or "tensorboard" if needed
        # learning_rate=1e-6,
        # num_generations=2,
        # per_device_train_batch_size=1,
        # temperature=1.7,
        # max_prompt_length=16,
        # gradient_accumulation_steps=1,
        # max_completion_length=8,
        # num_train_epochs=1,
        # gradient_checkpointing=True,
        # deepspeed=deepspeed_config_path,
    )

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int4_threshold=200.0)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32, trust_remote_code=True, quantization_config=quantization_config
    )

    lora_config = LoraConfig(
        r=2,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",  # ["o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    if policy_path is not None:
        print(f"Loading previous PEFT weights from {policy_path}")
        model = PeftModel.from_pretrained(model, policy_path, config=lora_config, is_trainable=True)
    else:
        print("No previous adapter found — starting fresh.")
        model = get_peft_model(model, lora_config)
        model.train()

    model.print_trainable_parameters()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

    # Train the model with GRPO
    trainer = GRPOTrainer(
        args=training_args,
        model=model,
        reward_funcs=rwd_fn.model,
        reward_processing_classes=rwd_fn.tokenizer,
        train_dataset=hf_dataset,
    )
    torch.cuda.empty_cache()
    trainer.train()

    # Save only the LoRA adapter
    model.save_pretrained(str(output_dir))
    del trainer, model, rwd_fn
    torch.cuda.empty_cache()


# def main(
#         dataset: pathlib.Path,
#         inference_prompt: pathlib.Path,
#         instruct_model: str,
#         checkpoints_dir: pathlib.Path,
#         model_dir: pathlib.Path
# ) -> None:
#     with open(inference_prompt, "r", encoding="utf-8") as file:
#         inference_prompt_template = escape_template(file.read())
#
#     with open(dataset, "r") as f:
#         train_dataset = TrainDataset.model_validate_json(f.read())
#
#     tlr_dataset = Dataset.from_dict({
#         "prompt": [
#             inference_prompt_template.format(input=i.prompt) for i in train_dataset.get_eligible_for_training()
#         ],
#         "chosen": [
#             i.chosen for i in train_dataset.get_eligible_for_training()
#         ],
#         "rejected": [
#             i.rejected for i in train_dataset.get_eligible_for_training()
#         ]
#     })
#
#     model = AutoModelForCausalLM.from_pretrained(
#         instruct_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
#     )
#
#     ref_model = AutoModelForCausalLM.from_pretrained(
#         instruct_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
#     )
#
#     tokenizer = AutoTokenizer.from_pretrained(instruct_model)
#
#     training_args = DPOConfig(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         max_grad_norm=0.3,
#         num_train_epochs=2,
#         learning_rate=5e-5,
#         save_total_limit=3,
#         logging_steps=10,
#         output_dir=checkpoints_dir,
#         # optim="paged_adamw_32bit",
#         lr_scheduler_type="cosine",
#         warmup_ratio=0.05,
#         remove_unused_columns=False,
#         bf16=True
#     )
#
#     dpo_trainer = DPOTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         beta=0.1,
#         train_dataset=tlr_dataset,
#         tokenizer=tokenizer,
#     )
#
#     dpo_trainer.train()
#
#     output_dir = pathlib.Path(model_dir)
#     dpo_trainer.save_model(output_dir)
#     dpo_trainer.model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # start method already set, safe to ignore
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
        "--max-interactions", default=15, type=int, help="Max number of student-teacher rounds"
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

    stf_dir = train_dir / "stf"

    if not stf_dir.exists():
        print(f" -------------------- ------------------ starting STF -------------------- ------------------")
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
        dataset = Dataset.load_from_disk(args.stf_dataset)


        def prepare_prompts(examples) -> None:
            _input = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                for c in examples["messages"]
            ]
            return {"text": _input}


        dataset = dataset.map(prepare_prompts, batched=True)

        training_args = SFTConfig(
            max_seq_length=1024,
            output_dir=stf_dir,
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

    textbooks = load_dataset("princeton-nlp/TextbookChapters")
    for i in range(args.num_iterations):
        print(f" -------------------- ------------------ starting it {i} -------------------- ------------------")
        train_it_dir = train_dir / f"iteration_{i}"
        train_it_dir.mkdir(exist_ok=True)

        seeds_path = train_it_dir / "seeds.json"
        interactions_path = train_it_dir / "interactions.json"
        evaluations_path = train_it_dir / "evaluations.json"
        action_vfn_model_dir = train_it_dir / "action_value_fn"
        value_checkpoints = train_it_dir / "value_checkpoints"
        policy_model_dir = train_it_dir / "policy_fn"
        policy_checkpoints = train_it_dir / "policy_checkpoints"

        previous_iteration = train_dir / f"iteration_{i - 1}"
        current_policy_path = previous_iteration / "policy_fn" if i > 0 else train_dir / "stf"
        current_vf_path = previous_iteration / "action_value_fn" if i > 0 else "allenai/longformer-base-4096"

        if policy_model_dir.exists():
            continue

        if not seeds_path.exists():
            seed_dataset = gen_seeds(textbooks, student, num_of_conversations=args.num_conversations)
            seeds_path.write_text(seed_dataset.model_dump_json(indent=4))

        if not interactions_path.exists():
            print()
            print("#### Rolling out policy")
            p = Process(
                target=rollout,
                args=(
                    seeds_path,
                    args.base_model,
                    current_policy_path,
                    interactions_path,
                    args.ollama_client,
                    args.max_interactions
                )
            )
            p.start()
            p.join()
            assert p.exitcode == 0
            p.close()

    #     if not evaluations_path.exists():
    #         print()
    #         print("#### Policy evaluation")
    #         interactions_dataset = InteractionDataset.model_validate_json(interactions_path.read_text())
    #         evaluations_dataset = evaluate(interactions_dataset, judge)
    #         evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))
    #
    #         print(f"\nAvg. perf. {i}: {evaluations_dataset.avg_performance()}\n", flush=True)
    #
    #     if not action_vfn_model_dir.exists():
    #         for j in range(args.vf_training_it):
    #             vf_training_path = train_it_dir / "vf_training"
    #             current_vf_step_path = current_vf_path if j == 0 else vf_training_path / f"it_{j - 1}" / "value_fn"
    #
    #             vf_target_path = (
    #                 vf_training_path / f"it_{j}" / "value_fn"
    #                 if j < args.vf_training_it - 1 else action_vfn_model_dir
    #             )
    #
    #             print()
    #             print("#### VF rollout")
    #             dataset_path = vf_training_path / f"it_{j}" / "dataset"
    #             p = Process(target=vf_rollout, args=(evaluations_path, str(current_vf_step_path), dataset_path, "cuda"))
    #             p.start()
    #             p.join()
    #             assert p.exitcode == 0
    #             p.close()
    #
    #             print()
    #             print("#### VF training")
    #             p = Process(
    #                 target=vf_train,
    #                 args=(dataset_path, value_checkpoints / f"it_{j}", str(current_vf_step_path), vf_target_path)
    #             )
    #             p.start()
    #             p.join()
    #             assert p.exitcode == 0
    #             p.close()
    #
    #     print()
    #     print("#### Policy training")
    #     print(f"current_policy_path={current_policy_path}")
    #     policy_train(
    #         evaluations_path,
    #         policy_checkpoints,
    #         str(action_vfn_model_dir),
    #         current_policy_path,
    #         policy_model_dir,
    #         base_model=(
    #             "microsoft/Phi-4-mini-instruct" if args.base_model == "phi4"
    #             else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    #         )
    #     )
