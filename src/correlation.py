import numpy as np
import pandas as pd
import os
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
PATH = os.getenv("PATH")

DATASETS = [
    {
        "name": "DETESTS",
        "csv_file": f"{PATH}/data/DETESTS.csv",
        "human_col": "human_majority_label",
        "llm_cols_to_compare": [
            "llms_consensus_label",
            "llms_majority_label",
            "llama_prediction",
            "nemo_prediction",
            "deepseek_prediction",
            "annotator1_label",
            "annotator2_label",
            "annotator3_label",
        ],
    },
    {
        "name": "HateXplain",
        "csv_file": f"{PATH}/data/HateXplain.csv",
        "human_col": "human_majority_label",
        "llm_cols_to_compare": [
            "llms_consensus_label",
            "llms_majority_label",
            "llama_prediction",
            "nemo_prediction",
            "deepseek_prediction",
            "annotator1_label",
            "annotator2_label",
            "annotator3_label",
        ],
    },
    {
        "name": "MHS",
        "csv_file": f"{PATH}/data/MHS.csv",
        "human_col": "human_majority_label",
        "llm_cols_to_compare": [
            "llms_consensus_label",
            "llms_majority_label",
            "llama_prediction",
            "nemo_prediction",
            "deepseek_prediction",
            "annotator1_label",
            "annotator2_label",
            "annotator3_label",
        ],
    },
]

flip_steps = np.arange(0, 1, 0.01)


def degrade_predictions(y_true, degradation_percent):
    y_pred = np.copy(y_true)
    n_flip = int(len(y_true) * degradation_percent)

    rs = np.random.RandomState(seed=int(degradation_percent * 10000) if degradation_percent > 0 else 42)
    flip_indices = rs.choice(len(y_true), size=n_flip, replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]
    return y_pred


def main():
    for dataset in DATASETS:
        print(f"\n=== Dataset: {dataset['name']} ===")
        df = pd.read_csv(dataset["csv_file"])
        gold = df[dataset["human_col"]].values.astype(int)

        degraded_systems = []
        for p in flip_steps:
            y_pred = degrade_predictions(gold, p)
            degraded_systems.append(y_pred)

        for llm_col in dataset["llm_cols_to_compare"]:
            llm = df[llm_col].values.astype(int)
            results = []
            for degraded_system in degraded_systems:
                f1_gt = f1_score(gold, degraded_system, average="binary")
                f1_llm = f1_score(llm, degraded_system, average="binary")
                results.append({"flip_percent": p, "f1_gold": f1_gt, "f1_llm": f1_llm})

            df_results = pd.DataFrame(results)
            df_results["f1_diff"] = df_results["f1_gold"] - df_results["f1_llm"]

            rank_gold = df_results["f1_gold"].rank(ascending=False)
            rank_llm = df_results["f1_llm"].rank(ascending=False)

            kendall_tau, _ = kendalltau(rank_gold, rank_llm)
            mean_f1_diff = df_results["f1_diff"].mean()

            print(f"  LLM: {llm_col}")
            print(f"    Kendall's tau: {kendall_tau:.3f}")
            print(f"    Mean F1 Difference (Gold - LLM): {mean_f1_diff:.3f}")


if __name__ == "__main__":
    main()
