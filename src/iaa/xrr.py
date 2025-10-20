from collections import Counter
from itertools import combinations
import os
import numpy as np
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env", override=True)
PATH = os.getenv("PATH")


def categorical_disagreement(x, y):
    return (x != y).astype(int)


def calculate_group_irr(A: np.ndarray) -> float:
    R = A.shape[1]
    if R < 2:
        return 1.0
    pair_kappas = []
    for r1_idx, r2_idx in combinations(range(R), 2):
        rater_r1 = A[:, [r1_idx]]
        rater_r2 = A[:, [r2_idx]]
        kappa = cohen_kappa_score(rater_r1, rater_r2)
        pair_kappas.append(kappa)
    return np.mean(pair_kappas)


def calculate_cross_kappa(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]  # Item num
    R = X.shape[1]  # Num annotations by item X
    S = Y.shape[1]  # Num annotations by item Y

    if n != Y.shape[0]:
        raise ValueError("X y Y must have the same number of items (rows).")

    total_observed_disagreement = 0
    for i in range(n):
        x_i = X[i, :]
        y_i = Y[i, :]
        disagreement_matrix_i = categorical_disagreement(
            x_i[:, np.newaxis],
            y_i[np.newaxis, :]
        )
        total_observed_disagreement += np.sum(disagreement_matrix_i)
    d_o = total_observed_disagreement / (n * R * S)

    all_x = X.flatten()
    all_y = Y.flatten()
    total_expected_disagreement = np.sum(
        categorical_disagreement(
            all_x[:, np.newaxis],
            all_y[np.newaxis, :]
        )
    )
    d_e = total_expected_disagreement / ((n ** 2) * R * S)

    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0

    kappa_x = 1.0 - (d_o / d_e)
    return kappa_x


def fleiss_kappa(A: np.ndarray) -> float:
    N = A.shape[0]  # Num items
    R = A.shape[1]  # Num raters

    if R < 2:
        return 1.0

    frequencies = np.zeros((N, 2))
    for i in range(N):
        counts = Counter(A[i, :])
        frequencies[i, 0] = counts.get(0, 0)
        frequencies[i, 1] = counts.get(1, 0)

    P_i = (np.sum(frequencies ** 2, axis=1) - R) / (R * (R - 1))
    P_bar = np.mean(P_i)
    p_j = np.sum(frequencies, axis=0) / (N * R)
    P_e_bar = np.sum(p_j ** 2)
    if P_e_bar == 1.0:
        return 1.0

    kappa_fleiss = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa_fleiss


if __name__ == "__main__":
    datasets = [
        {
            "name": "DETESTS",
            "csv": f"{PATH}/data/DETESTS-predictions.csv",
            "llm_label_cols": ['llama_prediction', 'nemo_prediction', 'deepseek_prediction'],
            "human_label_cols": ["annotator1_label", "annotator2_label", "annotator3_label"]
        },
        {
            "name": "HATEXPLAIN",
            "csv": f"{PATH}/data/HateXplain-predictions.csv",
            "llm_label_cols": ['llama_prediction', 'nemo_prediction', 'deepseek_prediction'],
            "human_label_cols": ["annotator1_label", "annotator2_label", "annotator3_label"]
        },
        {
            "name": "MHS",
            "csv": f"{PATH}/data/MHS-3ann-predictions.csv",
            "llm_label_cols": ['llama_prediction', 'nemo_prediction', 'deepseek_prediction'],
            "human_label_cols": ["annotator1_label", "annotator2_label", "annotator3_label"]
        }
    ]

    for ds in datasets:
        CSV_FILE_PATH = ds["csv"]
        HUMAN_COLS = ds["human_label_cols"]
        LLM_COLS = ds["llm_label_cols"]

        df = pd.read_csv(CSV_FILE_PATH)

        all_cols = HUMAN_COLS + LLM_COLS
        df[all_cols] = df[all_cols].astype(int)

        X = df[HUMAN_COLS].values
        Y = df[LLM_COLS].values

        # --- Cohen's Kappa pairwise ---
        irr_X_cohens = calculate_group_irr(X)
        irr_Y_cohens = calculate_group_irr(Y)

        # --- Cross kappa ---
        kappa_x = calculate_cross_kappa(X, Y)

        # --- Fleiss' kappa ---
        fleiss_human = fleiss_kappa(X)
        fleiss_llm = fleiss_kappa(Y)

        # --- Krippendorff's alpha ---
        alpha_human = krippendorff.alpha(reliability_data=X.T, level_of_measurement='nominal')
        alpha_llm = krippendorff.alpha(reliability_data=Y.T, level_of_measurement='nominal')

        irr_denominator = np.sqrt(irr_X_cohens) * np.sqrt(irr_Y_cohens)

        if irr_denominator <= 0:
            kappa_x_normalized = np.nan
            print("\nWARNING: IRR of one or both groups is not positive. Cannot be normalised with Cohen's Kappa.")
        else:
            kappa_x_normalized = kappa_x / irr_denominator
            print(f"dataset: {ds['name']}\nnormalized kappa: {kappa_x_normalized}")
            print(f"         kappa: {kappa_x}")
            print(f"         mean kappa human: {irr_X_cohens}")
            print(f"         mean kappa LLMs: {irr_Y_cohens}")
            print(f"Fleiss' alpha (humans): {fleiss_human}")
            print(f"Fleiss' alpha (LLMs): {fleiss_llm}")
            print(f"Krippendorff's alpha (humans): {alpha_human}")
            print(f"Krippendorff's alpha (LLMs): {alpha_llm}")
