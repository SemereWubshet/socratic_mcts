import argparse
import math
import pathlib
import time
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from ollama import Client
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer

from agents import Teacher, OllamaAgent, LLM
from mcts import discount_cumsum
from rollout import SeedDataset, gen_seeds, InteractionDataset, gen_teacher_student_interactions, EvaluationDataset, \
    evaluate, ChatHistory

import torch.multiprocessing as mp
from torch.multiprocessing import Process
from peft import LoraConfig


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


class SmolLM(LLM):

    def __init__(self,
                 base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                 max_length: int = 1024,
                 device: Optional[str] = None):
        self._model_name = base_model
        self.device = torch.device(device) if device is not None else None
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=torch.float16, trust_remote_code=True
        )

        if self.device is not None:
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.tokenizer.chat_template = (
            "{{ '<|im_start|>system\nFollow the Socratic method when answering to user queries.<|im_end|>\n' }}"
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def query(self, messages: List[Dict[str, str]]) -> str:
        raw_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = self.tokenizer(
            raw_prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        inputs = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask
        }

        if self.device is not None:
            inputs["input_ids"] = inputs["input_ids"].to(self.device)
            inputs["attention_mask"] = inputs["attention_mask"].to(self.device)

        output = self.model.generate(
            **inputs, max_new_tokens=250, temperature=1.7, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self._model_name

    def save(self, path: pathlib.Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class SimpleTeacher(Teacher):

    def chat(self, chat_history: ChatHistory) -> str:
        h = chat_history.format()
        return self._llm.query(h)

    def model_name(self) -> str:
        return self._llm.model_name


def rollout(
        dataset_path: pathlib.Path,
        policy_path: str,
        output_path: pathlib.Path,
        device: str,
        ollama_client: str,
        max_interactions: int
) -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)

    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(ollama_client))
    seed_dataset = SeedDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    smollm = SmolLM(policy_path, device=device)
    interactions_dataset = gen_teacher_student_interactions(
        seed_dataset, nemo, SimpleTeacher(smollm), max_interactions=max_interactions
    )
    output_path.write_text(interactions_dataset.model_dump_json(indent=4))

    torch.cuda.memory._dump_snapshot("/homes/mediouni/sources/socratic_mcts/pkl_files/memory_profile_rollout.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)


def vf_rollout(dataset_path: pathlib.Path, action_vf_path: str, output_path: pathlib.Path, device: str) -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)

    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    action_value_fn = ActionValueFn(action_vf_path, max_length=768, gpu=device)

    dataset = defaultdict(list)
    for item in evaluations_dataset.root:
        assessment = item.assessment
        history = item.interaction.chat_history.root
        trajectory = [ChatHistory.model_validate(history[:2 * z]).format() for z in range(1, math.ceil(len(history) / 2))]
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

    torch.cuda.memory._dump_snapshot("/homes/mediouni/sources/socratic_mcts/pkl_files/memory_profile_vf_rollout.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)


def vf_train(
        dataset_path: pathlib.Path,
        value_checkpoints_path: pathlib.Path,
        action_value_fn_path: str,
        vf_output_path: pathlib.Path
) -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)

    tokenized_dataset = Dataset.load_from_disk(dataset_path)
    training_args = TrainingArguments(output_dir=value_checkpoints_path, num_train_epochs=1, learning_rate=1e-5, gradient_checkpointing=True)
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=768)
    trainer = Trainer(model=action_value_fn.model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    action_value_fn.save(vf_output_path)

    torch.cuda.memory._dump_snapshot("/homes/mediouni/sources/socratic_mcts/pkl_files/memory_profile_vf_train.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)


def policy_train(
        dataset_path: pathlib.Path,
        policy_checkpoints: pathlib.Path,
        action_value_fn_path: str,
        policy_path: str,
        output_dir: pathlib.Path
) -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)

    evaluations_dataset = EvaluationDataset.model_validate_json(pathlib.Path(dataset_path).read_text())
    dataset = defaultdict(list)
    for item in evaluations_dataset.root:
        history = item.interaction.chat_history.root
        trajectory = [ChatHistory.model_validate(history[:2 * z + 1]).format() for z in range(math.ceil(len(history) / 2))]
        dataset["prompt"].extend(trajectory)

    hf_dataset = Dataset.from_dict(dataset)

    rwd_fn = ActionValueFn(action_value_fn_path, max_length=768)
    deepspeed_config_path = "/homes/mediouni/sources/socratic_mcts/src/deepspeed_config2.json"
    training_args = GRPOConfig(
        output_dir=policy_checkpoints,
        learning_rate=1e-6,
        num_generations=2,
        per_device_train_batch_size=1,
        temperature=1.7,
        max_prompt_length=16,
        gradient_accumulation_steps=1,
        max_completion_length=8,
        num_train_epochs=1,
        gradient_checkpointing=True,
        # deepspeed=deepspeed_config_path,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        args=training_args,
        model=policy_path,
        reward_funcs=rwd_fn.model,
        reward_processing_classes=rwd_fn.tokenizer,
        train_dataset=hf_dataset,
        peft_config=lora_config,
    )
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
    trainer.train()
    trainer.save_model(output_dir)

    torch.cuda.memory._dump_snapshot("/homes/mediouni/sources/socratic_mcts/pkl_files/memory_profile_policy_train.pkl")
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FULL RL", description="Train a socratic llm through RL."
    )

    parser.add_argument("--root-dir", required=True, type=pathlib.Path, help="Path where to store pipeline artifacts")
    parser.add_argument("--num-iterations", required=True, type=int, help="Number of training iterations")
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
    parser.add_argument("--train-batch-size", type=int, default=1, help="Batch size for training")

    args = parser.parse_args()

    mp.set_start_method('spawn')

    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")

    train_dir: pathlib.Path = args.root_dir / "train"
    train_dir.mkdir(exist_ok=True, parents=True)

    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(args.ollama_client))

    llama3 = OllamaAgent(model="llama3.3", client=Client(args.ollama_client))

    for i in range(args.num_iterations):
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
        current_policy_path = previous_iteration / "policy_fn" if i > 0 else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        current_vf_path = previous_iteration / "action_value_fn" if i > 0 else "allenai/longformer-base-4096"

        if policy_model_dir.exists():
            continue

        if not seeds_path.exists():
            seed_dataset = gen_seeds(wikipedia, nemo, num_of_conversations=args.num_conversations)
            seeds_path.write_text(seed_dataset.model_dump_json(indent=4))
        print(f"interactions path:{interactions_path.exists()}")
        if not interactions_path.exists():
            p = Process(
                target=rollout,
                args=(seeds_path, str(current_policy_path), interactions_path, "cuda", args.ollama_client, args.max_interactions)
            )
            p.start()
            p.join()
            assert p.exitcode == 0
            p.close()

        if not evaluations_path.exists():
            interactions_dataset = InteractionDataset.model_validate_json(interactions_path.read_text())
            evaluations_dataset = evaluate(interactions_dataset, llama3)
            evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))

            print(f"\nAvg. perf. {i}: {evaluations_dataset.avg_performance()}\n", flush=True)

        if not action_vfn_model_dir.exists():
            for j in range(args.vf_training_it):
                vf_training_path = train_it_dir / "vf_training"
                current_vf_step_path = current_vf_path if j == 0 else vf_training_path / f"it_{j - 1}" / "value_fn"

                vf_target_path = (
                    vf_training_path / f"it_{j}" / "value_fn"
                    if j < args.vf_training_it - 1 else action_vfn_model_dir
                )

                dataset_path = vf_training_path / f"it_{j}" / "dataset"
                p = Process(target=vf_rollout, args=(evaluations_path, str(current_vf_step_path), dataset_path, "cuda"))
                p.start()
                p.join()
                assert p.exitcode == 0
                p.close()

                p = Process(
                    target=vf_train,
                    args=(dataset_path, value_checkpoints / f"it_{j}", str(current_vf_step_path), vf_target_path)
                )
                p.start()
                p.join()
                assert p.exitcode == 0
                p.close()

        print("Starting policy training...")
        policy_train(
            evaluations_path,
            policy_checkpoints,
            str(action_vfn_model_dir),
            str(current_policy_path),
            policy_model_dir
        )
