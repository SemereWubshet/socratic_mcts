import argparse
import pathlib
from typing import Dict, List, Tuple


# from agents import StudentSeed, LLM, Student, Teacher, Judge, OllamaAgent
from rollout import Interaction, InteractionDataset, Evaluation, EvaluationDataset

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
    mistral_nemo_path = output_dir / "eval_mistral_nemo.json"

    # Read the evaluation dataset
    empty_eval_path = pathlib.Path(args.empty_eval)
    if not empty_eval_path.exists():
        print("Input evaluation dataset not found.", flush=True)
        exit(1)

    print("Loading empty evaluation dataset", flush=True)

    empty_eval_dataset = EvaluationDataset.model_validate_json(empty_eval_path.read_text())
    interaction_datasets = extract_interaction_dataset(empty_eval_dataset)
    interaction_path.write_text(interaction_datasets.model_dump_json(indent=4))




    # Now you can access each interaction dataset in the list
    # for interaction_dataset in interaction_datasets:
    #   print(interaction_dataset)
    #
    # evaluation_dataset = load_evaluation_dataset(evaluation_path.read_text())
    #
    # # Iterate through models and generate evaluations for each
    # for model_name in args.models:
    #     model_output_dir = output_dir / model_name
    #     model_output_dir.mkdir(exist_ok=True)
    #
    #     print(f"Evaluating dataset with model: {model_name}", flush=True)
    #
    #     # Evaluate using the current model
    #     model_evaluations = evaluate_model(evaluation_dataset, model_name)
    #
    #     # Write the evaluated dataset to file
    #     model_output_file = model_output_dir / "evaluations.json"
    #     model_output_file.write_text(model_evaluations.model_dump_json(indent=4))
    #
    # print("Evaluation process complete for all models.")
    # exit(0)