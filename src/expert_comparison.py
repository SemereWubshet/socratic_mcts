import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from evaluate import ResultDataset


def bootstrap_ci(y_true, y_pred, n_bootstraps=1000):
    kappas = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bs_true = y_true[indices]
        bs_pred = y_pred[indices]
        kappa = cohen_kappa_score(bs_true, bs_pred)
        kappas.append(kappa)

    kappas = np.array(kappas)
    valid_kappas = kappas[~np.isnan(kappas)]
    if len(valid_kappas) == 0:
        return np.nan, np.nan

    return np.percentile(valid_kappas, [2.5, 97.5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expert", nargs=2, type=pathlib.Path,
                        help="Path to the evaluation dataset with human assessments and feedback.")
    args = parser.parse_args()

    with open(args.expert[0], "r") as f:
        expert_1 = ResultDataset.model_validate_json(f.read())

    df1 = pd.DataFrame(
        {
            "question_id": [e.id for e in expert_1.evaluations],
            "assessment": [e.assessment for e in expert_1.evaluations]
        }
    )
    df1 = df1.set_index("question_id")

    with open(args.expert[1], "r") as f:
        expert_2 = ResultDataset.model_validate_json(f.read())

    df2 = pd.DataFrame(
        {
            "question_id": [e.id for e in expert_2.evaluations],
            "assessment": [e.assessment for e in expert_2.evaluations]
        }
    )
    df2 = df2.set_index("question_id")

    join = df1.join(df2, how="inner", on="question_id", lsuffix="_1", rsuffix="_2")

    y1 = join["assessment_1"].to_numpy(dtype=int)
    y2 = join["assessment_2"].to_numpy(dtype=int)

    cm = confusion_matrix(y1, y2, labels=[0, 1])
    kappa = cohen_kappa_score(y1, y2)
    ci_lower, ci_upper = bootstrap_ci(y1, y2)

    print("Confusion Matrix:")
    print(cm)
    print(f"Cohens kappa: {kappa} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print()
