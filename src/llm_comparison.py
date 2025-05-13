import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from evaluate import ResultDataset


def parse(dataset: ResultDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [dataset.model_name] * len(dataset.evaluations),
            "question_id": [e.id for e in dataset.evaluations],
            "assessment": [e.assessment for e in dataset.evaluations]
        }
    )


def bootstrap_ci(y_true, y_pred, n_bootstraps=10000):
    kappas = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bs_true = y_true[indices]
        bs_pred = y_pred[indices]
        kappa = cohen_kappa_score(bs_true, bs_pred, labels=[0, 1])
        kappas.append(kappa)

    kappas = np.array(kappas)
    kappas = kappas[~np.isnan(kappas)]
    if len(kappas) == 0:
        return np.nan, np.nan

    kde = gaussian_kde(kappas, bw_method=0.1)
    kde_samples = kde.resample(10000)
    return np.percentile(kde_samples, [2.5, 97.5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dataset", type=pathlib.Path, required=True,
                        help="Path to the evaluation dataset with human assessments and feedback.")
    parser.add_argument("--evaluation-dir", type=pathlib.Path, required=True,
                        help="Directory to comparisons for each model.")
    args = parser.parse_args()

    with open(args.reference_dataset, "r") as f:
        results_dataset = ResultDataset.model_validate_json(f.read())

    human = parse(results_dataset)
    human = human.set_index("question_id")

    df = pd.DataFrame()
    for evaluation in args.evaluation_dir.glob("*.json"):
        with open(evaluation, "r") as f:
            eval_dataset = ResultDataset.model_validate_json(f.read())
        df = pd.concat([df, parse(eval_dataset)], ignore_index=True)

    results = []
    for model_name, table in df.groupby("model_name"):
        join = human.join(
            table.set_index("question_id"),
            how="inner",
            on="question_id",
            lsuffix="_h",
            rsuffix="_llm"
        )
        join = join[["assessment_h", "assessment_llm"]]
        join = join.dropna(subset=["assessment_llm"])

        y_true = join["assessment_h"].to_numpy(dtype=int)
        y_pred = join["assessment_llm"].to_numpy(dtype=int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1])
        ci_lower, ci_upper = bootstrap_ci(y_true, y_pred)

        # Parse model name and size
        model_parts = model_name.split(":", 1)
        model_name_clean = model_parts[0]
        model_size = model_parts[1] if len(model_parts) > 1 else "N/A"

        results.append({
            "model_name": model_name_clean,
            "model_size": model_size,
            "cohens_kappa": kappa,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

        print(f"Model: {model_name}")
        print("Confusion Matrix:")
        print(cm)
        print(f"Cohens kappa: {kappa} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
        print()

    # Create and sort summary table
    summary_df = pd.DataFrame(results)
    summary_df["CI"] = summary_df.apply(
        lambda x: f"[{x['ci_lower']:.3f}, {x['ci_upper']:.3f}]", axis=1
    )
    summary_df = summary_df.sort_values(["model_name", "model_size"])
    summary_df = summary_df[["model_name", "model_size", "cohens_kappa", "CI"]]

    print("\nFinal Summary Table:")
    print(summary_df.to_string(index=False))
