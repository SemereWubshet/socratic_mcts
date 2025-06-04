# Debug helper
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct")

for name, module in model.named_modules():
    if "W" in name or "proj" in name:
        print(name)
