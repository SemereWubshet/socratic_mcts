import argparse
import pathlib
from os import mkdir

import ollama
import json
from openai import OpenAI

import shutil
from typing import Dict, List, Tuple

from pydantic import BaseModel

from agents import StudentSeed, LLM, Student, Teacher, Judge, OllamaAgent, OpenAIAgent
from rollout import Interaction, InteractionDataset, Evaluation, EvaluationDataset, evaluate

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from statistics import mean, stdev


class JudgeEvalDataset(BaseModel):
    model: str
    analysis: list[Evaluation]


# Extract interaction dataset from an evaluation dataset
def extract_interaction_dataset(evaluation_dataset: EvaluationDataset) -> InteractionDataset:
  all_interactions = InteractionDataset([])
  for evaluation in evaluation_dataset.root:
    interaction = evaluation.interaction
    all_interactions.root.append(interaction)
  return all_interactions

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the evaluation dataset with human assessments and feedback.",
                        default="caches/human-eval.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store evaluation datasets for each model.",
                        default="datasets")
    parser.add_argument("--use_cache", type=bool, required=False,
                        help="Enable to utilize cached evaluations",
                        default=False)
    parser.add_argument("--ollama_address", type=str, required=False,
                        help="The address for ollama server.",
                        default="http://atlas1api.eurecom.fr:8019")
    args = parser.parse_args()

    # Check args
    if not pathlib.Path(args.input_file).exists():
        print(f"Input file does not exist!", flush=True)
        exit(1)

    if not pathlib.Path(args.output_dir).exists():
        print("Output folder does not exist. \nCreating folder")
        output_path = pathlib.Path(args.output_dir)
        mkdir(output_path)

    # Prepare LLM list
    ollama_list = ["deepseek-v2:16b"] # ["llama3.3:70b", "mistral-nemo:12b-instruct-2407-fp16"]
    openai_list = [] # ["gpt-4o"]

    # Setup output directory
    output_dir = args.output_dir

    # Read the evaluation dataset
    human_eval_path = pathlib.Path(args.input_file)
    if not human_eval_path:
        print("Input evaluation dataset not found.", flush=True)
        exit(1)

    print("Loading human evaluation dataset", flush=True)
    json_data = json.loads(human_eval_path.read_text())
    data = json.dumps(json_data['analysis'], ensure_ascii=False, indent=4)
    human_eval_dataset = EvaluationDataset.model_validate_json(data)
    ollama_dataset_list = []
    openai_dataset_list = []

    if not args.use_cache:
        print("No cache - Loading interactions dataset", flush=True)
        interactions_dataset = extract_interaction_dataset(human_eval_dataset)

        for llm_name in ollama_list + openai_list:
            clean_llm_name = llm_name.replace(":", ".")
            llm_path = pathlib.Path(output_dir + f"/eval_{clean_llm_name}.json")

            if llm_name in ollama_list:
                llm_judge = OllamaAgent(model=llm_name, client=ollama.Client(args.ollama_address), temperature=0.)
            else:
                llm_judge = OpenAIAgent(model=llm_name, client=OpenAI(), temperature=0.)

            print(f"\nEvaluating {llm_name}", flush=True)
            analysis = evaluate(interactions_dataset, llm_judge)
            print(f"Finished evaluation for {llm_name}\nWriting file into {llm_path}")

            judge_eval = JudgeEvalDataset(model=llm_name, analysis=analysis.root)
            llm_path.write_text(judge_eval.model_dump_json(indent=4))


    if args.use_cache: print("Cached LLM evaluations being loaded", flush=True)

    human_assessment = [evaluation.assessment for evaluation in human_eval_dataset.root] # human assessment as ground truth

    cache_dir = pathlib.Path(args.output_dir)
    evals = {}
    for child in cache_dir.iterdir():
        judge_eval = JudgeEvalDataset.model_validate_json(child.read_text())
        evals[judge_eval.model] = {}
        evals[judge_eval.model]["assessment"] = [evaluation.assessment for evaluation in judge_eval.analysis]
        evals[judge_eval.model]["kappa"] = cohen_kappa_score(human_assessment, evals[judge_eval.model]["assessment"])

    # Print only the model and its kappa score
    for model, data in evals.items():
        print(f"{model}, Kappa Score: {data['kappa']}")

