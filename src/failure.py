import glob
import pathlib
import random

from transformers import pipeline

from schemas import EvaluationDataset

if __name__ == "__main__":
    input_dir = pathlib.Path("./datasets/tmp")

    failures = []

    for name in glob.glob(str(input_dir / "eval_*.json")):
        with open(name) as f:
            dataset = EvaluationDataset.model_validate_json(f.read())
            if dataset.metadata.max_interactions >= 16:
                for e in dataset.evaluations:
                    if not e.assessment:
                        # needs better code here
                        summary = e.feedback.split("# Verdict Summary")[-1].strip()
                        failures.append(summary)

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    labels = [
        "Skipped Key Topics",
        "Too Superficial",
        "Missed Guiding Questions",
        "Failed to Connect Ideas",
        "Didn’t Adapt to Student",
        "Other Issue"
    ]

    out = classifier(failures, labels, multi_label=False)
    print(out)
