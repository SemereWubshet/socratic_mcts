import gc
import gc
import json
import pathlib
from typing import List, Dict, Any

import torch
import unsloth
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer

from agents import LLM


class Qwen(LLM):

    def __init__(self, base_model: str, max_length: int = 1024):
        self._base_model = base_model
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> str:
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
            load_in_8bit=False,
        )

        if for_inference:
            self.model = unsloth.FastLanguageModel.for_inference(self.model)

        # ✅ Patch apply_chat_template to default enable_thinking=False
        if hasattr(self.tokenizer, "apply_chat_template"):
            original_fn = self.tokenizer.apply_chat_template

            def patched_apply_chat_template(conversation, **kwargs):
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


def policy_train(
        dataset_path: pathlib.Path,
        policy_path: pathlib.Path,
        checkpoints_dir: pathlib.Path,
        output_dir: pathlib.Path
) -> Dict[str, Any]:
    dict_train_dataset = json.loads(dataset_path.read_text(encoding="UTF-8"))
    cleaned = {
        "prompt": dict_train_dataset["prompt"],
        "chosen": dict_train_dataset["chosen"],
        "rejected": dict_train_dataset["rejected"]
    }
    train_dataset = Dataset.from_dict(cleaned)
    train_dataset = train_dataset.shuffle()

    qwen = Qwen(str(policy_path))
    qwen.load(for_inference=False)

    training_args = DPOConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        max_length=1024,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=checkpoints_dir,
        overwrite_output_dir=True,
        max_prompt_length=128,
    )

    dpo_trainer = DPOTrainer(
        model=qwen.model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=qwen.tokenizer
    )

    results = dpo_trainer.train()
    qwen.save(output_dir)
    qwen.unload()

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
        per_device_train_batch_size=4,
        num_train_epochs=2.,
        output_dir=train_dir / "stf",
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

    with open("/home/gatti/socratic-rl/trial-Z/train/dpo_dataset.json", "r") as f:
        d = json.loads(f.read())

    d_dpo_dataset = {"prompt": [], "chosen": [], "rejected": []}

    minmaxs = []
    for e in d["evaluations"]:
        scores = []
        for i in e["eval"]:
            scores.append(i[1])
        minmax = max(scores) - min(scores)
        if minmax > 0.05:
            d_dpo_dataset["prompt"].append(e["prompt"])
            d_dpo_dataset["chosen"].append(e["chosen"])
            d_dpo_dataset["rejected"].append(e["rejected"])

    print(len(d_dpo_dataset["prompt"]))

    with open("/home/gatti/socratic-rl/trial-Z.1/dpo_dataset.json", "w") as f:
        json.dump(d_dpo_dataset, f)

    d = policy_train(
        pathlib.Path("/home/gatti/socratic-rl/trial-Z.1/dpo_dataset.json"),
        pathlib.Path("/home/gatti/socratic-rl/trial-Z/train/stf/pretrained/"),
        pathlib.Path("/home/gatti/socratic-rl/trial-Z.1/checkpoints/"),
        pathlib.Path("/home/gatti/socratic-rl/trial-Z.1/policy_fn/"),
    )

    print(d)

    pathlib.Path("/home/gatti/socratic-rl/trial-Z.1/stats.json").write_text(json.dumps(d))
