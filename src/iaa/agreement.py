import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap
import itertools

load_dotenv(dotenv_path=".env", override=True)
PATH = os.getenv("PATH")
CSV_FILE_PATH = f"{PATH}/data/<dataset_name>.csv"

HUMAN_COLS = ['annotator1_label', 'annotator2_label', 'annotator3_label']
LLM_COLS = ['llama_prediction', 'nemo_prediction', 'deepseek_prediction']
GROUP_COLS = ['human_majority', 'human_consensus', 'llms_majority', 'llms_consensus']

COLUMN_NAMES = {
    'annotator1_label': 'Human 1',
    'annotator2_label': 'Human 2',
    'annotator3_label': 'Human 3',
    'human_majority': 'Human majority',
    'human_consensus': 'Human consensus',
    'llama_prediction': 'Llama 3.1',
    'nemo_prediction': 'Nemo',
    'deepseek_prediction': 'DeepSeek',
    'llms_majority': 'LLMs majority',
    'llms_consensus': 'LLMs consensus'
}

cmap_custom = LinearSegmentedColormap.from_list("custom_gradient", ["#f9e1e8", "#d2067a"])


def majority_vote(df, cols):
    """Compute majority vote (2/3) across given columns."""
    votes = df[cols].astype(int)
    return (votes.sum(axis=1) > (len(cols) / 2)).astype(int)


def main():
    df = pd.read_csv(CSV_FILE_PATH)

    # Recompute majority/consensus dynamically
    df['human_majority'] = majority_vote(df, HUMAN_COLS)
    df['human_consensus'] = (df[HUMAN_COLS].sum(axis=1) == len(HUMAN_COLS)).astype(int)  # full consensus
    df['llms_majority'] = majority_vote(df, LLM_COLS)
    df['llms_consensus'] = (df[LLM_COLS].sum(axis=1) == len(LLM_COLS)).astype(int)  # full consensus

    all_cols = HUMAN_COLS + LLM_COLS + GROUP_COLS
    df[all_cols] = df[all_cols].astype(int)

    # --- 1. Pairwise κ (humans, LLMs, cross) ---
    def pairwise_kappa(cols):
        pairs = list(itertools.combinations(cols, 2))
        kappas = []
        for a, b in pairs:
            k = cohen_kappa_score(df[a], df[b])
            kappas.append((f"{a}-{b}", k))
        return kappas

    print("\nHuman pairwise κ:")
    human_kappas = pairwise_kappa(HUMAN_COLS)
    for name, k in human_kappas:
        print(f"  {name}: {k:.3f}")
    print(f"  → Mean inter-human κ: {np.mean([k for _, k in human_kappas]):.3f}")

    print("\nLLM pairwise κ:")
    llm_kappas = pairwise_kappa(LLM_COLS)
    for name, k in llm_kappas:
        print(f"  {name}: {k:.3f}")
    print(f"  → Mean inter-LLM κ: {np.mean([k for _, k in llm_kappas]):.3f}")

    print("\nHuman–LLM pairwise κ:")
    cross_pairs = list(itertools.product(HUMAN_COLS, LLM_COLS))
    cross_kappas = []
    for a, b in cross_pairs:
        k = cohen_kappa_score(df[a], df[b])
        cross_kappas.append((f"{a}-{b}", k))
        print(f"  {a} vs {b}: {k:.3f}")
    print(f"  → Mean human–LLM κ: {np.mean([k for _, k in cross_kappas]):.3f}")

    # --- 2. Leave-one-out κ (within group) ---
    def loo_kappa(cols):
        kappas = []
        for a in cols:
            others = [x for x in cols if x != a]
            gold_minus = majority_vote(df, others)
            k = cohen_kappa_score(df[a], gold_minus)
            kappas.append((a, k))
        return kappas

    print("\nHuman leave-one-out κ:")
    loo_human = loo_kappa(HUMAN_COLS)
    for a, k in loo_human:
        print(f"  {a} vs gold(-{a}): {k:.3f}")
    print(f"  → Mean leave-one-out human κ: {np.mean([k for _, k in loo_human]):.3f}")

    print("\nLLM leave-one-out κ:")
    loo_llm = loo_kappa(LLM_COLS)
    for a, k in loo_llm:
        print(f"  {a} vs gold(-{a}): {k:.3f}")
    print(f"  → Mean leave-one-out LLM κ: {np.mean([k for _, k in loo_llm]):.3f}")

    # --- 3. Cross group κ ---
    llm_majority = majority_vote(df, LLM_COLS)
    human_majority = majority_vote(df, HUMAN_COLS)

    print("\nCross κ (Human vs. LLM majority):")
    human_vs_llm = []
    for a in HUMAN_COLS:
        k = cohen_kappa_score(df[a], llm_majority)
        human_vs_llm.append(k)
        print(f"  {a} vs LLM majority: {k:.3f}")
    print(f"  → Mean Human–LLM majority κ: {np.mean(human_vs_llm):.3f}")

    print("\nCross κ (LLM vs. Human majority):")
    llm_vs_human = []
    for a in LLM_COLS:
        k = cohen_kappa_score(df[a], human_majority)
        llm_vs_human.append(k)
        print(f"  {a} vs Human majority: {k:.3f}")
    print(f"  → Mean LLM–Human majority κ: {np.mean(llm_vs_human):.3f}")

    # --- 3.1. Cross LLM vs Human leave-one-out κ ---
    print("\nCross LLM vs Human leave-one-out κ:")
    for llm in LLM_COLS:
        for leave_out in HUMAN_COLS:
            others = [h for h in HUMAN_COLS if h != leave_out]
            gold_minus = majority_vote(df, others)  # LOO gold
            k = cohen_kappa_score(df[llm], gold_minus)
            print(f"  {llm} vs gold(-{leave_out}): {k:.3f}")

    # --- 4. Group-level κ ---
    group_kappa = cohen_kappa_score(human_majority, llm_majority)
    print(f"\nGroup-level (human majority vs LLM majority) κ: {group_kappa:.3f}")
    group_consensus_kappa = cohen_kappa_score(df['human_consensus'], df['llms_consensus'])
    print(f"Group-level (human consensus vs LLM consensus) κ: {group_consensus_kappa:.3f}")

    # --- 5. Heatmap ---
    heatmap_cols = HUMAN_COLS + LLM_COLS
    kappa_matrix = pd.DataFrame(index=heatmap_cols, columns=heatmap_cols, dtype=float)
    for r1 in heatmap_cols:
        for r2 in heatmap_cols:
            if r1 == r2:
                kappa_matrix.loc[r1, r2] = 1.0
            else:
                kappa_matrix.loc[r1, r2] = round(cohen_kappa_score(df[r1], df[r2]), 2)


    kappa_matrix = kappa_matrix.rename(index=COLUMN_NAMES, columns=COLUMN_NAMES)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        kappa_matrix,
        annot=True,
        cmap=cmap_custom,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={'size': 28},
        cbar_kws={'label': "Cohen’s κ"}
    )
    plt.title("MHS", fontsize=32, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=24)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=24)
    plt.tight_layout()
    output_path = f"{PATH}/results/<dataset_name>_pairwise_kappa_heatmap.pdf"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    print(f"\nHeatmap saved to {output_path}")
    print(kappa_matrix.round(3))


if __name__ == "__main__":
    main()
