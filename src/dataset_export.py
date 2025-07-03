from pathlib import Path

from datasets import Dataset

from schemas import EvaluationDataset

if __name__ == "__main__":
    evaluation_dataset_path = Path("../datasets/evaluation/eval_16_mistral-small3.1_24b.json")
    eval_dataset = EvaluationDataset.model_validate_json(evaluation_dataset_path.read_text())

    output = {"messages": []}
    for e in filter(lambda _e: _e.assessment, eval_dataset.evaluations):
        formatted = []
        for h in e.interaction.chat_history.root:
            role = "user" if h.role == "Student" else "assistant"
            formatted.append({"role": role, "content": h.content})
        output["messages"].append(formatted)

    dataset = Dataset.from_list(output)
    dataset.save_to_disk("../datasets/stf_examples")
