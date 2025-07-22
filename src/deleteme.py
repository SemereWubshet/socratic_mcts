import pathlib

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel, AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def init_model(path_to_dir: pathlib.Path) -> None:
    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path="answerdotai/ModernBERT-large",
        num_labels=1,
        torch_dtype=torch.float32,
        problem_type="regression",
        device_map="cuda"
    )

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
    base_model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=4,
        lora_alpha=32,
        task_type=TaskType.SEQ_CLS,
        target_modules="all-linear"
    )
    model = get_peft_model(base_model, peft_config)

    print(model.modules_to_save)

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
