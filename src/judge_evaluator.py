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
    parser.add_argument("--use_cache", type=bool, required=True,
                        help="Decides whether or not to use cached judgements.")
    parser.add_argument("--num_eval", type=bool, required=True,
                        help="The number of times evaluations are made from which statistics are recorded.")
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

    print("Evaluating interactions as we speak", flush=True)
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

