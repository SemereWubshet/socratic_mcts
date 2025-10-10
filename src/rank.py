from scipy.stats import spearmanr, kendalltau
import pandas as pd
import json


if __name__ == "__main__":
    with open("/homes/gatti/sources/socratic_mcts/datasets/judge-benchmark/qwen3_32b.json") as f:
        qwen = json.load(f)
    
    qwen_values = [e["assessment"] for e in sorted(qwen["evaluations"], key=lambda e: e["id"])]

    with open("/homes/gatti/sources/socratic_mcts/datasets/judge-benchmark/gemma3_27b.json") as f:
        gemma = json.load(f)

    gemma_values = [e["assessment"] for e in sorted(gemma["evaluations"], key=lambda e: e["id"])]

    df = pd.DataFrame({
        "gemma": gemma_values,
        "qwen": qwen_values
    })
    rho, p_rho = spearmanr(df["gemma"], df["qwen"])
    tau, p_tau = kendalltau(df["gemma"], df["qwen"])

    agreement = (df["gemma"] == df["qwen"]).mean()
    print(f"Agreement = {agreement:.2%}")
    print(f"Spearman ρ = {rho:.3f} (p={p_rho:.3g}), Kendall τ = {tau:.3f} (p={p_tau:.3g})")
