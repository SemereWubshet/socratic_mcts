import itertools
import json
from pathlib import Path

from datasets import Dataset

from schemas import EvaluationDataset

if __name__ == "__main__":
    input_dir = Path("../datasets/")
    output_dir = Path("../datasets/socratic_traces")
    messages = []
    output_dir.mkdir(parents=True)

    traces = (output_dir / "traces.jsonl")
    traces.touch()
    raw = traces.open("a", encoding="UTF-8")

    for json_file in itertools.chain(
            input_dir.glob("evaluation/eval_16_mistral-small3.1_24b.json"),
            input_dir.glob("evaluation_300_0/eval_*.json"),
            input_dir.glob("evaluation_300_1/eval_*.json"),
            input_dir.glob("evaluation_300_2/eval_*.json"),
            input_dir.glob("evaluation_600_3/eval_*.json"),
            input_dir.glob("evaluation_1000_4/eval_*.json"),
    ):
        eval_dataset = EvaluationDataset.model_validate_json(json_file.read_text())

        for e in eval_dataset.evaluations:
            raw.write(json.dumps({"metadata": eval_dataset.metadata.model_dump(), "evaluation": e.model_dump()}) + "\n")

            if e.assessment:
                formatted = []
                for h in e.interaction.chat_history.root[:-1]:
                    role = "user" if h.role == "Student" else "assistant"
                    formatted.append({"role": role, "content": h.content})
                messages.append({"messages": formatted})

    raw.close()

    # Now build a single Dataset from all messages combined
    dataset = Dataset.from_list(messages)

    # Save the combined dataset
    dataset.save_to_disk(output_dir / "positive_traces")

    print(f"Processed {len(messages)} conversations from {len(list(input_dir.glob('eval_*.json')))} files.")
