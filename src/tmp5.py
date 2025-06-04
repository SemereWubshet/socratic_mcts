import torch
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.to(torch.device("cuda"))

