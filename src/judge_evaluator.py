import argparse
import pathlib
import ollama
from typing import Dict, List, Tuple

from openai import OpenAI

from agents import StudentSeed, LLM, Student, Teacher, Judge, OllamaAgent, OpenAIAgent
from rollout import Interaction, InteractionDataset, Evaluation, EvaluationDataset, evaluate

def extract_interaction_dataset(evaluation_dataset: EvaluationDataset) -> InteractionDataset:

  all_interactions = InteractionDataset([])
  for evaluation in evaluation_dataset.root:
    interaction = evaluation.interaction
    all_interactions.root.append(interaction)
  return all_interactions

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty_eval", type=str, required=True,
                        help="Path to the evaluation dataset with empty assessments and feedback.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store evaluation datasets for each model.")
    args = parser.parse_args()

    # Setup output directory
    output_dir = pathlib.Path(args.output_dir)
    interaction_path = output_dir / "interaction.json"
    ollama_path = output_dir / "eval_ollama.json"
    openai_path = output_dir / "eval_openai.json"
    mistralnemo_path = output_dir / "eval_mistralnemo.json"

    # Read the evaluation dataset
    empty_eval_path = pathlib.Path(args.empty_eval)
    if not empty_eval_path.exists():
        print("Input evaluation dataset not found.", flush=True)
        exit(1)

    print("Loading empty evaluation dataset", flush=True)
    empty_eval_dataset = EvaluationDataset.model_validate_json(empty_eval_path.read_text())

    print("Loading interactions dataset", flush=True)
    interactions_dataset = extract_interaction_dataset(empty_eval_dataset)
    interaction_path.write_text(interactions_dataset.model_dump_json(indent=4))

    # List of AI agents
    ollama_judge = OllamaAgent(model="llama3.3:70b", client=ollama.Client("http://atlas1api.eurecom.fr:8019"))
    # mistralnemo_judge = OllamaAgent(model="mistral-nemo:12b-instruct-2407-fp16", client=ollama.Client("http://atlas1api.eurecom.fr:8019"))
    # openai_judge = OpenAIAgent(model="gpt-4o-mini", client=OpenAI()) # Change to gpt 4o

    # Evaluate dataset by the 3 judges
    print("Evaluating interactions as we speak", flush=True)
    ollama_eval_dataset = evaluate(interactions_dataset, ollama_judge)
    ollama_path.write_text(ollama_eval_dataset.model_dump_json(indent=4))

    # mistralnemo_eval_dataset = evaluate(interactions_dataset, mistralnemo_judge)
    # openai_eval_dataset = evaluate(interactions_dataset, openai_judge)

