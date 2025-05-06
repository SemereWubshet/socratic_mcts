import pandas as pd
from sklearn.metrics import cohen_kappa_score

from evaluate import ResultDataset
from rollout import EvaluationDataset

if __name__ == "__main__":
    expert_1: EvaluationDataset
    with open("datasets/expert-1.json", "r") as f:
        expert_1 = ResultDataset.model_validate_json(f.read())

    expert_2: EvaluationDataset
    with open("datasets/expert-2.json", "r") as f:
        expert_2 = ResultDataset.model_validate_json(f.read())

    y1 = []
    y2 = []

    good_good = 0
    good_bad = 0
    bad_good = 0
    bad_bad = 0
    for i in expert_1.evaluations:
        for j in expert_2.evaluations:
            if i.id == j.id and i.assessment is not None and j.assessment is not None:
                y1.append(i.assessment)
                y2.append(j.assessment)

                if i.assessment and j.assessment:
                    good_good += 1
                elif i.assessment and not j.assessment:
                    good_bad += 1
                    print(f"id: {i.id}")
                    print(f"main topics:\n{i.interaction.seed.main_topics}")
                    print(f"expert-1: {i.assessment}")
                    print(f"{i.feedback}")
                    print()
                    print(f"expert-2: {j.assessment}")
                    print(f"{j.feedback}")
                    print()
                elif not i.assessment and j.assessment:
                    bad_good += 1
                    print(f"id: {i.id}")
                    print(f"main topics:\n{i.interaction.seed.main_topics}")
                    print(f"expert-1: {i.assessment}")
                    print(f"{i.feedback}")
                    print()
                    print(f"expert-2: {j.assessment}")
                    print(f"{j.feedback}")
                    print()
                else:
                    bad_bad += 1
                break

    conf_matrix = pd.DataFrame(
        [[good_good, good_bad],
         [bad_good, bad_bad]],
        columns=["Rater 2 - Positive", "Rater 2 - Negative"],
        index=["Rater 1 - Positive", "Rater 1 - Negative"]
    )

    # Print the confusion matrix elegantly
    print("Confusion Matrix:")
    print(conf_matrix)

    print()
    print(f"Cohens kappa: {cohen_kappa_score(y1, y2)}")
