import argparse
import math
import pathlib
from collections import defaultdict
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset, Dataset
from ollama import Client
from peft import LoraConfig, PeftModel, get_peft_model
from torch.multiprocessing import Process
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from agents import Teacher, OllamaAgent, LLM
from mcts import discount_cumsum
from rollout import SeedDataset, gen_seeds, InteractionDataset, gen_teacher_student_interactions, EvaluationDataset, \
    evaluate, ChatHistory


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
    SYSTEM_PROMPT = ("You are a Socratic tutor. Use the following principles in responding to students:\n"
                     "  - Ask thought-provoking, open-ended questions that challenge students' preconceptions and "
                     "encourage them to engage in deeper reflection and critical thinking.\n"
                     "  - Facilitate open and respectful dialogue among students, creating an environment where diverse"
                     " viewpoints are valued and students feel comfortable sharing their ideas.\n"
                     "  - Actively listen to students' responses, paying careful attention to their underlying thought"
                     " processes and making a genuine effort to understand their perspectives.\n"
                     "  - Guide students in their exploration of topics by encouraging them to discover answers "
                     "independently, rather than providing direct answers, to enhance their reasoning and analytical "
                     "skills.\n"
                     "  - Promote critical thinking by encouraging students to question assumptions, evaluate "
                     "evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n"
                     "  - Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a "
                     "growth mindset and exemplifying the value of lifelong learning.\n"
                     "  - Keep interactions short, limiting yourself to one question at a time and to concise "
                     "explanations.")

    def __init__(self,
                 base_model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                 adapter_path: Optional[pathlib.Path] = None,
                 max_length: int = 1024,
                 device: Optional[str] = None):
        self._model_name = base_model
        self.device = torch.device(device) if device is not None else None
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if self.device is not None:
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self.tokenizer.chat_template = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
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
            **inputs, max_new_tokens=128, do_sample=True, temperature=0.15, pad_token_id=self.tokenizer.eos_token_id
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


class Phi4(LLM):
    SYSTEM_PROMPT = ("You are a Socratic tutor. Use the following principles in responding to students:\n"
                     "  - Ask thought-provoking, open-ended questions that challenge students' preconceptions and "
                     "encourage them to engage in deeper reflection and critical thinking.\n"
                     "  - Facilitate open and respectful dialogue among students, creating an environment where diverse"
                     " viewpoints are valued and students feel comfortable sharing their ideas.\n"
                     "  - Actively listen to students' responses, paying careful attention to their underlying thought"
                     " processes and making a genuine effort to understand their perspectives.\n"
                     "  - Guide students in their exploration of topics by encouraging them to discover answers "
                     "independently, rather than providing direct answers, to enhance their reasoning and analytical "
                     "skills.\n"
                     "  - Promote critical thinking by encouraging students to question assumptions, evaluate "
                     "evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n"
                     "  - Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a "
                     "growth mindset and exemplifying the value of lifelong learning.\n"
                     "  - Keep interactions short, limiting yourself to one question at a time and to concise "
                     "explanations.")

    def __init__(self,
                 base_model: str = "microsoft/Phi-4-mini-instruct",
                 adapter_path: Optional[pathlib.Path] = None,
                 max_length: int = 1024,
                 device: Optional[str] = None):
        self._model_name = base_model
        self.device = torch.device(device) if device is not None else None
        self.max_length = max_length
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int4_threshold=200.0)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=torch.float32, trust_remote_code=True, quantization_config=quantization_config
        )
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if self.device is not None:
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def query(self, messages: List[Dict[str, str]]) -> str:
        _messages = [{"role": "system", "content": self.SYSTEM_PROMPT.strip()}, ] + messages
        raw_prompt = self.tokenizer.apply_chat_template(_messages, tokenize=False, add_generation_prompt=True)
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
            **inputs, max_new_tokens=128, do_sample=True, temperature=0.15
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
        base_model: str,
        policy_path: Optional[pathlib.Path],
        output_path: pathlib.Path,
        device: str,
        ollama_client: str,
        max_interactions: int
) -> None:
    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(ollama_client))
    seed_dataset = SeedDataset.model_validate_json(pathlib.Path(dataset_path).read_text())

    if base_model == "phi4":
        model = Phi4(adapter_path=policy_path, device=device)
    else:
        model = SmolLM(adapter_path=policy_path, device=device)

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

    args = parser.parse_args()

    mp.set_start_method('spawn')

    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")

    train_dir: pathlib.Path = args.root_dir / "train"
    train_dir.mkdir(exist_ok=True, parents=True)

    nemo = OllamaAgent(model="mistral-small3.1:24b", client=Client(args.ollama_client))

    llama3 = OllamaAgent(model="mistral-small3.1:24b", client=Client(args.ollama_client))

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
        current_policy_path = previous_iteration / "policy_fn" if i > 0 else None
        current_vf_path = previous_iteration / "action_value_fn" if i > 0 else "allenai/longformer-base-4096"

        if policy_model_dir.exists():
            continue

        if not seeds_path.exists():
            seed_dataset = gen_seeds(wikipedia, nemo, num_of_conversations=args.num_conversations)
            seeds_path.write_text(seed_dataset.model_dump_json(indent=4))

        if not interactions_path.exists():
            print()
            print("#### Rolling out policy")
            p = Process(
                target=rollout,
                args=(seeds_path, args.base_model, current_policy_path, interactions_path, "cuda", args.ollama_client,
                      args.max_interactions)
            )
            p.start()
            p.join()
            assert p.exitcode == 0
            p.close()

        if not evaluations_path.exists():
            print()
            print("#### Policy evaluation")
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

                print()
                print("#### VF rollout")
                dataset_path = vf_training_path / f"it_{j}" / "dataset"
                p = Process(target=vf_rollout, args=(evaluations_path, str(current_vf_step_path), dataset_path, "cuda"))
                p.start()
                p.join()
                assert p.exitcode == 0
                p.close()

                print()
                print("#### VF training")
                p = Process(
                    target=vf_train,
                    args=(dataset_path, value_checkpoints / f"it_{j}", str(current_vf_step_path), vf_target_path)
                )
                p.start()
                p.join()
                assert p.exitcode == 0
                p.close()

        print()
        print("#### Policy training")
        print(f"current_policy_path={current_policy_path}")
        policy_train(
            evaluations_path,
            policy_checkpoints,
            str(action_vfn_model_dir),
            current_policy_path,
            policy_model_dir,
            base_model=(
                "microsoft/Phi-4-mini-instruct" if args.base_model == "phi4"
                else "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            )
        )
