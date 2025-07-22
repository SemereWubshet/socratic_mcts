import argparse
import gc
import json
import math
import pathlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import scipy
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForSequenceClassification


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
        load_pretrained = pathlib.Path(self._base_model).exists() and pathlib.Path(self._base_model).is_dir()
        if load_pretrained:
            config = ModernBertConfig.from_pretrained(self._base_model)
        else:
            config = ModernBertConfig(
                classifier_activation="tanh",
                classifier_bias=True,
                classifier_dropout=0.05
            )

        self.model = ModernBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self._base_model,
            config=config,
            num_labels=1,
            torch_dtype=torch.float32,
            problem_type="regression",
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model)

        if not load_pretrained:
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

        for name, param in self.model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item():.4f}")

    def unload(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


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


def vf_rollout(
        input_dataset: List[Dict[str, Any]],
        action_vf_path: str,
        _lambda: float,
        output_path: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    action_value_fn = ActionValueFn(action_vf_path, max_length=2048)

    all_preds = []
    all_targets = []
    dataset = defaultdict(list)
    for item in tqdm(input_dataset, desc="vf rollout"):
        assessment: bool = item["assessment"]
        history: List[Dict[str, str]] = item["messages"]
        trajectory = [
            [
                {"role": h["role"], "content": h["content"]} for h in history[:2 * z]
            ] for z in range(1, math.ceil(len(history) / 2))
        ]
        values = [float(action_value_fn(h)) for h in trajectory]

        gamma = 1.
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

    if output_path is not None:
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
        device: str
) -> Dict[str, Any]:
    tokenized_dataset = Dataset.load_from_disk(dataset_path)
    tokenized_dataset = tokenized_dataset.shuffle()
    training_args = TrainingArguments(
        num_train_epochs=1,
        output_dir=value_checkpoints_path,
        overwrite_output_dir=True,
        learning_rate=1e-5,
        gradient_checkpointing=True
    )
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=2048)
    action_value_fn.load()
    trainer = Trainer(model=action_value_fn.model, args=training_args, train_dataset=tokenized_dataset)
    results = trainer.train()
    action_value_fn.save(vf_output_path)

    action_value_fn.unload()

    return results.metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-traces", default=Path("./datasets/socratic_traces/traces.jsonl"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vf-training-it", default=10, type=int)
    parser.add_argument("--lambda", dest="lambda_", default=0.3, type=float)
    parser.add_argument("--clean", action="store_true", help="Clean root_dir if it exists")
    args = parser.parse_args()

    train_dir: Path = args.output_dir

    if args.clean and train_dir.exists() and train_dir.is_dir():
        for child in train_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    train_dir.mkdir(parents=True, exist_ok=True)

    traces = args.input_traces.read_text().split("\n")

    messages = []
    for trace in traces:
        if not trace:
            continue
        data = json.loads(trace)
        chat_history: List[Dict[str, str]] = data['evaluation']['interaction']['chat_history']
        formatted = []
        for h in chat_history:
            role = "user" if h["role"] == "Student" else "assistant"
            formatted.append({"role": role, "content": h["content"]})
        messages.append(
            {"messages": formatted, "assessment": data["evaluation"]["assessment"]}
        )

    print(f"Processed {len(messages)} conversations.")

    random.shuffle(messages)
    split = math.ceil(len(messages) * 0.9)
    # TODO: rebalance accept/reject test dataset?
    train, test = messages[:split], messages[split:]

    vf_training_path = train_dir / "vf_training"

    action_value_fn = ActionValueFn("answerdotai/ModernBERT-large", max_length=2048)
    action_value_fn.load()
    action_value_fn([{"role": "user", "content": "What's the capital of Brazil?"},
                     {"role": "assistant", "content": "Brasília!"}])
    action_value_fn.save(vf_training_path / "it_0" / "value_fn")
    action_value_fn.unload()
    del action_value_fn

    stats = {}

    print()
    print("#### VF training")
    stats["vf_training"] = {"vf_loss": [], "explained_var": [], "train": [], "min": [], "max": []}
    stats["vf_eval"] = {"vf_loss": [], "explained_var": [], "min": [], "max": []}
    for j in range(1, args.vf_training_it + 1):
        current_vf_step_path = vf_training_path / f"it_{j - 1}" / "value_fn"
        vf_target_path = vf_training_path / f"it_{j}" / "value_fn"
        dataset_path = vf_training_path / f"it_{j}" / "dataset"

        print()
        print(f"dataset_path={dataset_path}")
        print(f"current_vf_step_path={current_vf_step_path}")
        print(f"vf_target_path={vf_target_path}")
        print()

        d = vf_rollout(train, str(current_vf_step_path), args.lambda_, dataset_path)
        stats["vf_training"]["vf_loss"].append(d["vf_loss"])
        stats["vf_training"]["explained_var"].append(d["explained_var"])
        stats["vf_training"]["min"].append(d["min"])
        stats["vf_training"]["max"].append(d["max"])

        d = vf_rollout(test, str(current_vf_step_path), args.lambda_)
        stats["vf_eval"]["vf_loss"].append(d["vf_loss"])
        stats["vf_eval"]["explained_var"].append(d["explained_var"])
        stats["vf_eval"]["min"].append(d["min"])
        stats["vf_eval"]["max"].append(d["max"])

        d = vf_train(
            dataset_path,
            str(current_vf_step_path),
            vf_target_path,
            train_dir / "vf_checkpoints",
            device="cuda"
        )

        stats["vf_training"]["train"].append(d)

        print()
        print(json.dumps(stats))
        print()
