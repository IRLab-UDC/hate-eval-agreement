# 🧩 Can LLMs Evaluate What They Cannot Annotate? Revisiting Reliability in Hate Speech Detection


## 📑 Citation

Soon! 🚀


## 🗂️ Project Structure


```markdown
.
├── data/
│   ├── DETESTS.csv         # DETESTS dataset and predictions
│   ├── HateXplain.tsv      # HateXplain dataset and predictions
│   └── MHS.tsv             # MHS dataset and predictions (only rows with 3 annotators)
│
├── src/
│   ├── iaa/
│   │   ├── agreement.py    # Computes Cohen’s κ, pairwise, leave one out, cross-dataset, etc.
│   │   └── xrr.py          # Computes cross-Rater Reliability (xRR), Fleiss’ κ, and Krippendorff’s α
│   │
│   ├── correlation.py      # Ranking correlation experiment (Kendall’s τ)
│   └── inference.py        # Generates LLM-based judgements for each dataset
│
└── README.md
```


## ⚙️ Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Set your environment variables for LLM API access:

```bash
PATH=<your_path>
HF_TOKEN=<your_hf_token>
```

3. Run the experiments using the provided scripts.


## 🔮 Generating LLM Judgements

We use instruction-tuned LLMs to automatically label social media messages as hate speech (`True`) or non-hate (`False`).

This script performs batched inference with consistent prompts, model-specific chat templates, and strict output control to ensure binary predictions.

### ⚙️ Usage

Run the inference script to generate predictions from any supported model:

```bash
python src/inference.py \
  --dataset data/<dataset_name>.csv \
  --model <model> \
  --output <output_file_name>
```

#### Arguments

| Argument | Description |
|-----------|-------------|
| `--dataset` | Path to input dataset (CSV/TSV) containing at least columns `id` and `text`. |
| `--model` | One of `llama`, `nemo`, or `deepseek`. |
| `--output` | Name of the output file (default: `predictions.tsv`). |
| `--start_id` | Optional row index to resume interrupted runs. |


## 🧮 Agreement Analysis

Compute inter-annotator agreement metrics:

```bash
python src/iaa/agreement.py
```

```bash
python src/iaa/xrr.py
```

### Metrics

- Cohen’s κ — Pairwise agreement corrected for chance.

- Fleiss’ κ — Generalisation for multiple annotators.

- Krippendorff’s α — Robust to missing data and metric scales.

- xRR — Cross-rater reliability, modelling systematic disagreement patterns.

Outputs include per-dataset summaries and visualisations of agreement distributions.


## 📊 Ranking Correlation Experiment

Evaluate LLMs as evaluators instead of annotators, testing whether they can replicate human-based model rankings.

```bash
python src/correlation.py
```

### Procedure

- Simulate classifiers by randomly flipping $p%$ of oracle predictions.

- Evaluate each synthetic classifier under human and LLM labels.

- Compute Kendall’s τ to measure ranking consistency.

- Report mean absolute F1 differences to quantify score distortion.



## ⚠️ Disclaimer

This repository includes research content that may involve hate speech or offensive language.
Such content is used solely for research and analysis purposes in the context of hate speech detection.
The authors and maintainers do not endorse or promote any hateful or discriminatory views.

Users are advised to handle the data responsibly and in compliance with ethical guidelines.


## 🙏 Acknowledgements

Soon! 🚀


## 📜 License

This project is licensed under the Apache License 2.0 – see the LICENSE
 file for details.


## 📬 Contact

Soon! 🚀