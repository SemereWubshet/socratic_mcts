import argparse
import pathlib
import ollama
from openai import OpenAI

import shutil
from typing import Dict, List, Tuple
# from fsspec.caching import caches


from agents import StudentSeed, LLM, Student, Teacher, Judge, OllamaAgent, OpenAIAgent
from rollout import Interaction, InteractionDataset, Evaluation, EvaluationDataset, evaluate

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean, stdev

# Extract interaction dataset from an evaluation dataset
def extract_interaction_dataset(evaluation_dataset: EvaluationDataset) -> InteractionDataset:
  all_interactions = InteractionDataset([])
  for evaluation in evaluation_dataset.root:
    interaction = evaluation.interaction
    all_interactions.root.append(interaction)
  return all_interactions

# Get assessments from evaluation lists
def extract_assessments(evaluation_list):
    return [evaluation.assessment for evaluation in evaluation_list.root]


# Function to calculate metrics
def calculate_metrics(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_eval", type=str, required=True,
                        help="Path to the evaluation dataset with human assessments and feedback.",
                        default="caches/human-eval.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store evaluation datasets for each model.",
                        default="datasets")
    parser.add_argument("--use_cache", type=str, required=False,
                        help="If using cached evaluation files, provide the directory containing them.",
                        default=False)
    parser.add_argument("--num_eval", type=int, required=False,
                        help="The number of times evaluations are made from which statistics are recorded.",
                        default="2")
    args = parser.parse_args()

    # Prepare LLM list
    ollama_list = ["llama3.3:70b"] # "mistral-nemo:12b-instruct-2407-fp16"]
    # Mistral doesn't work because it is not 'smart' enough to give a proper json response
    openai_list = ["gpt-4o"]

    # Setup output directory
    output_dir = pathlib.Path(args.output_dir)
    interaction_path = output_dir / "interaction.json"

    # Read the evaluation dataset
    human_eval_path = pathlib.Path(args.human_eval)
    if not human_eval_path:
        print("Input evaluation dataset not found.", flush=True)
        exit(1)

    print("Loading human evaluation dataset", flush=True)
    human_eval_dataset = EvaluationDataset.model_validate_json(human_eval_path.read_text())
    ollama_dataset_list = []
    openai_dataset_list = []

    if not args.use_cache:
        print("No cache - Loading interactions dataset", flush=True)
        interactions_dataset = extract_interaction_dataset(human_eval_dataset)
        interaction_path.write_text(interactions_dataset.model_dump_json(indent=4))

        # Prepare LLM dictionary
        llm_dict = {}
        # for llm in ollama_list:
        #     clean_llm = llm.replace(":", ".")
        #     llm_dict[llm] = {"judge": OllamaAgent(model=llm, client=ollama.Client("http://atlas1api.eurecom.fr:8019")),
        #                           "path": args.output_dir + f"/ollama/eval_{clean_llm}"}

        for llm in openai_list:
            clean_llm = llm.replace(":", ".")
            llm_dict[llm] = {"judge": OpenAIAgent(model=llm, client=OpenAI()),
                                  "path": args.output_dir + f"/openai/eval_{clean_llm}"}

        for i in range(args.num_eval):
            for llm in llm_dict:
                print(f"Evaluating for {i}th {llm}", flush=True)
                llm_dict[llm][f"eval_dataset_{i}"] = evaluate(interactions_dataset, llm_dict[llm]["judge"])
                print(f"Finished evaluation for {llm}\nWriting file {i}")

                # Add dataset to list
                if llm in ollama_list: ollama_dataset_list.append(llm_dict[llm][f"eval_dataset_{i}"])
                if llm in openai_list: openai_dataset_list.append(llm_dict[llm][f"eval_dataset_{i}"])

                file_path = pathlib.Path(llm_dict[llm]["path"] + f"_{i}.json")
                file_path.write_text(llm_dict[llm][f"eval_dataset_{i}"].model_dump_json(indent=4))

    if args.use_cache:
        print("Cached LLM evaluations being loaded", flush=True)
        caches_dir = pathlib.Path(args.use_cache)
        ollama_dir = caches_dir / "ollama"
        openai_dir = caches_dir / "openai"

        for child in ollama_dir.iterdir():
            if child.is_file():
                ollama_dataset_list.append(EvaluationDataset.model_validate_json(child.read_text()))
        for child in openai_dir.iterdir():
            if child.is_file():
                openai_dataset_list.append(EvaluationDataset.model_validate_json(child.read_text()))
        print("LLM evaluations loaded", flush=True)

    # Extract assessments
    ollama_assessments = [extract_assessments(dataset) for dataset in ollama_dataset_list]
    openai_assessments = [extract_assessments(dataset) for dataset in openai_dataset_list]
    human_assessments = extract_assessments(human_eval_dataset)

    # Compare Ollama evaluations with ground truth
    ollama_metrics = [calculate_metrics(assessment, human_assessments) for assessment in ollama_assessments]
    ollama_mean = [mean(column) for column in zip(*ollama_metrics)]
    # ollama_stdev = [stdev(column) for column in zip(*ollama_metrics)]

    print(ollama_mean)
    # print(ollama_stdev)

    # Compare Openai evaluations with ground truth

    openai_metrics = [calculate_metrics(assessment, human_assessments) for assessment in openai_assessments]
    openai_mean = [mean(column) for column in zip(*openai_metrics)]
    # openai_stdev = [stdev(column) for column in zip(*openai_metrics)]

    print(openai_mean)
    # print(openai_stdev)
























    # Print results
    # print(ollama_metrics)
    # print(openai_metrics)
    print("Ollama Model Metrics:")
    # print(f"Accuracy: {ollama_metrics[0]:.2f}")
    # print(f"Precision: {ollama_metrics[1]:.2f}")
    # print(f"Recall: {ollama_metrics[2]:.2f}")
    # print(f"F1 Score: {ollama_metrics[3]:.2f}")


    # print("\nOpenAI Model Metrics:")
    # print(f"Accuracy: {openai_metrics[0]:.2f}")
    # print(f"Precision: {openai_metrics[1]:.2f}")
    # print(f"Recall: {openai_metrics[2]:.2f}")
    # print(f"F1 Score: {openai_metrics[3]:.2f}")
























    # Ollama
    # ollama_judge = OllamaAgent(model="llama3.3:70b", client=ollama.Client("http://atlas1api.eurecom.fr:8019"))
    # ollama_eval_dataset = evaluate(interactions_dataset, ollama_judge)
    # ollama_path.write_text(ollama_eval_dataset.model_dump_json(indent=4))

    # Mistral
    # mistralnemo_judge = OllamaAgent(model="mistral-nemo:12b-instruct-2407-fp16", client=ollama.Client("http://atlas1api.eurecom.fr:8019"))
    # mistralnemo_eval_dataset = evaluate(interactions_dataset, mistralnemo_judge)
    # mistralnemo_path.write_text(mistralnemo_eval_dataset.model_dump_json(indent=4))

    # OpenAI
    # openai_judge = OpenAIAgent(model="gpt-4o-mini", client=OpenAI()) # Change to gpt 4o
    # openai_eval_dataset = evaluate(interactions_dataset, openai_judge)
    # openai_path.write_text(openai_eval_dataset.model_dump_json(indent=4))

