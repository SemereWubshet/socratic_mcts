import argparse
import gc
import json
import math
import pathlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import scipy
import torch
from datasets import Dataset
from torch import nn
from torch.nn import MSELoss
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, ModernBertConfig, \
    ModernBertPreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead


class ActionValueFunctionModel(ModernBertPreTrainedModel):
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
        self.model = ActionValueFunctionModel.from_pretrained(
            pretrained_model_name_or_path=self._base_model,
            config=ModernBertConfig(
                num_labels=1,
                torch_dtype=torch.bfloat16,
                hidden_size=1024,
                use_flash_attention_2=False
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model)
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
        if self.device is not None:
            self.model = self.model.to(self.device)

        for name, param in self.model.named_parameters():
            if name == "classifier.bias":
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
        output_path: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    action_value_fn = ActionValueFn(action_vf_path, max_length=2048, gpu="cuda")

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
    explained_var = float(1 - np.var(all_targets - all_preds) / (np.var(all_targets) + 1e-8))
    _min = np.min(all_preds)
    _max = np.max(all_preds)

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
    action_value_fn = ActionValueFn(action_value_fn_path, max_length=2048, gpu=device)
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
    parser.add_argument("--clean", action="store_true", help="Clean root_dir if it exists")
    args = parser.parse_args()

    train_dir = args.output_dir

    if args.clean:
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

    action_value_fn = ActionValueFn("answerdotai/ModernBERT-large", max_length=2048, gpu="cuda")
    action_value_fn.load()
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

        d = vf_rollout(train, str(current_vf_step_path), dataset_path)
        stats["vf_training"]["vf_loss"].append(d["vf_loss"])
        stats["vf_training"]["explained_var"].append(d["explained_var"])
        stats["vf_training"]["min"].append(d["min"])
        stats["vf_training"]["max"].append(d["max"])

        d = vf_rollout(test, str(current_vf_step_path))
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
