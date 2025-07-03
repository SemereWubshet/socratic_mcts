from pathlib import Path

from datasets import Dataset

from schemas import EvaluationDataset

if __name__ == "__main__":
    input_dir = Path("../datasets/evaluation")
    messages = []

    for json_file in input_dir.glob("eval_*.json"):
        eval_dataset = EvaluationDataset.model_validate_json(json_file.read_text())

        for e in filter(lambda _e: _e.assessment, eval_dataset.evaluations):
            formatted = []
            for h in e.interaction.chat_history.root[:-1]:
                role = "user" if h.role == "Student" else "assistant"
                formatted.append({"role": role, "content": h.content})
            messages.append({"messages": formatted})

    # Now build a single Dataset from all messages combined
    dataset = Dataset.from_list(messages)

    # Save the combined dataset
    dataset.save_to_disk("../datasets/stf_examples_combined")

    print(f"Processed {len(messages)} conversations from {len(list(input_dir.glob('eval_*.json')))} files.")
