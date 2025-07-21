import argparse
import pathlib

from socratic_rl_peft import Qwen

if __name__ == "__main__":
    # model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    #     # model_name="unsloth/Qwen3-4B",
    #     model_name="/home/gatti/socratic-rl/trial-0/train/stf/checkpoint-246/",
    #     dtype=torch.bfloat16,
    #     max_seq_length=1024,
    #     load_in_4bit=False,  # False for LoRA 16bit
    #     load_in_8bit=False,
    # )
    # model = unsloth.FastLanguageModel.for_inference(model)
    # # tokenizer = unsloth.get_chat_template(tokenizer, chat_template="qwen3")
    #
    # question = [{"role": "user", "content": "What's the capital of Brazil?"}, ]
    #
    # chat = tokenizer.apply_chat_template(
    #     question, tokenize=False, add_generation_prompt=True, enable_thinking=False
    # )
    # print(chat)
    # inputs = tokenizer([chat], return_tensors="pt").to("cuda")
    # print(f"Inputs:\n{inputs}")
    #
    # outputs = model.generate(
    #     **inputs, max_new_tokens=128, do_sample=True, temperature=0.15
    # )
    # print(f"Outputs:\n{outputs}")
    # print("\n\n")
    # generation = outputs[0, len(inputs['input_ids'][0]):]
    # print(f"Generation:\n{generation}")
    # print("\n\n")
    # decoded = tokenizer.decode(generation, skip_special_tokens=True)
    # print(f"decoded: '{decoded}'")

    parser = argparse.ArgumentParser()
    parser.add_argument("POLICY_PATH", type=pathlib.Path)
    parser.add_argument("--question", type=str, default="What's the capital of Brazil?")
    args = parser.parse_args()

    qwen = Qwen(base_model=str(args.POLICY_PATH))
    response = qwen.query([{"role": "user", "content": args.question}, ])
    print(response)
