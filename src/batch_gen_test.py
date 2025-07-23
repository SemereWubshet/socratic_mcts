import argparse
import pathlib

from socratic_rl_peft import Qwen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("BASE_MODEL", type=pathlib.Path)
    args = parser.parse_args()

    llm = Qwen(str(args.BASE_MODEL))

    print("Q: What's the capital of Brazil?")

    messages = [
        [
            {"role": "user", "content": "What's the capital of Brazil?"},
            {"role": "assistant", "content": "What are big cities in Brazil?"},
            {"role": "user", "content": "Rio, São Paulo, Brasíla, Porto Alegre..."}
        ],
        [
            {"role": "user", "content": "What's the capital of Brazil?"}
        ]
    ]
    raw_prompts = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    print(raw_prompts)

    inputs = llm.tokenizer(raw_prompts, padding="max_length", padding_side="left", return_tensors="pt").to("cuda")

    print(inputs)

    outputs = llm.model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.
    )

    decoded = llm.tokenizer.decode(outputs, skip_special_tokens=True)

    print(decoded)
