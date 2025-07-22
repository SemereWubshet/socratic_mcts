import pathlib
from typing import Union, Optional

import pathlib
from typing import Union, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, AutoPeftModelForSequenceClassification
from torch.nn import MSELoss
from transformers import AutoTokenizer, AutoModel, ModernBertConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead, \
    ModernBertForSequenceClassification, \
    ModernBertPreTrainedModel


class ActionValueFunctionModel(ModernBertPreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = False

    def __init__(self, encoder: PreTrainedModel, config: ModernBertConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_labels = config.num_labels
        self.config = config

        self.model = encoder
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)

        self.activation = nn.GELU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.value = nn.Tanh()

        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module):
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

        if isinstance(module, ModernBertForSequenceClassification):
            init_weight(module.classifier, self.config.hidden_size ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

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


def init_model(path_to_dir: pathlib.Path) -> None:
    # base_model = AutoModelForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path="answerdotai/ModernBERT-large",
    #     num_labels=1,
    #     torch_dtype=torch.float32,
    #     problem_type="regression",
    #     device_map="cuda"
    # )

    encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-large")

    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(["[USER]", "[/USER]", "[EOT]"])
    tokenizer.chat_template = (
        "{% for i in range(0, messages|length, 2) %}"
        "{% if i + 1 < messages|length %}"
        "[USER]{{ messages[i].content }}[/USER] {{ messages[i+1].content }}[EOT]\n"
        "{% endif %}"
        "{% endfor %}"
    )
    encoder.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules="all-linear"
    )
    encoder = get_peft_model(encoder, peft_config)

    model = ActionValueFunctionModel(encoder, ModernBertConfig(vocab_size=len(tokenizer)), device_map="cuda")

    print(list(model.modules()))

    # merged = model.merge_and_unload()

    # Save LoRA adapter
    model.save_pretrained(path_to_dir)

    # Save tokenizer
    tokenizer.save_pretrained(path_to_dir)


def reload_model(path_to_dir: pathlib.Path) -> None:
    print("reloading")
    tokenizer = AutoTokenizer.from_pretrained(path_to_dir)

    # base_model = AutoModelForSequenceClassification.from_pretrained(
    #     str(path_to_dir),
    #     torch_dtype=torch.float32,
    #     device_map="cuda"
    # )
    #
    # for name, param in base_model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item():.4f}")

    config = PeftConfig.from_pretrained(str(path_to_dir))
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        # base_model,
        path_to_dir,
        is_trainable=True,
        device_map="cuda"
    )

    print()
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, mean={param.data.mean().item():.4f}")


if __name__ == "__main__":
    init_model(pathlib.Path("/home/gatti/test"))
    reload_model(pathlib.Path("/home/gatti/test"))
