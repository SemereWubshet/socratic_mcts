import argparse
import pathlib
from os import mkdir
from typing import List

from ollama import Client
from openai import OpenAI
from tqdm import tqdm

from agents import OllamaAgent, OpenAIAgent, Judge
from evaluate import ResultDataset
from rollout import Evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=pathlib.Path, required=True,
                        help="Path to the evaluation dataset with human assessments and feedback.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True,
                        help="Directory to store evaluation datasets for each model.")
    parser.add_argument("--ollama-address", type=str, required=True, help="The address for ollama server.")
    parser.add_argument("--models", nargs="+", default=[
        "llama3.3:70b", "phi4:14b", "mistral-small3.1:24b", "gemma3:27b",
        "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b", "qwen3:32b",
        "gpt-4o", "gpt-4o-mini"
    ])
    args = parser.parse_args()

    # Check args
    if not args.dataset_file.exists():
        print(f"Input file does not exist!", flush=True)
        exit(1)

    if not args.output_dir.exists():
        print("Output folder does not exist. \nCreating folder")
        mkdir(args.output_dir)

    with open(args.dataset_file, "r") as f:
        result_dataset = ResultDataset.model_validate_json(f.read())

    openai_client = OpenAI()
    ollama_client = Client(args.ollama_address)

    ollama_client.ps()
    openai_models = [m.id for m in openai_client.models.list().data]

    llms = []
    for model in args.models:
        if model in openai_models:
            llms.append(OpenAIAgent(model, openai_client, temperature=0.15))
        else:
            llms.append(OllamaAgent(model, ollama_client, num_ctx=5120, temperature=0.15))

    for llm in tqdm(llms, desc="LLMs"):
        judge = Judge(llm)

        evaluations: List[Evaluation] = []
        for e in tqdm(result_dataset.evaluations):
            feedback, assessment = judge.evaluate(e.interaction.seed.main_topics, str(e.interaction.chat_history))
            evaluations.append(
                Evaluation(id=e.id, interaction=e.interaction, feedback=feedback, assessment=assessment)
            )
        judge_eval = ResultDataset(model_name=llm.model_name, evaluations=evaluations)

        with open(args.output_dir / (llm.model_name.replace(":", "_") + ".json"), "w") as f:
            f.write(judge_eval.model_dump_json(indent=4))
