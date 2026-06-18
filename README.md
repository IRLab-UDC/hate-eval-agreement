# 🧩 Can LLMs Evaluate What They Cannot Annotate? Revisiting Reliability in Hate Speech Detection

This work was accepted at the Fifteenth biennial Language Resources and Evaluation Conference (LREC 2026). Preprint [here](https://arxiv.org/abs/2512.09662)! 📄

## 📑 Citation

```bash
@inproceedings{piot-etal-2026-can,
  title = {Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection},
  author = {Piot, Paloma and Otero, David and Martin-Rodilla, Patricia and Parapar, Javier},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {4358--4370},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/22n5hekovvrz},
  abstract = {Hate speech spreads widely online and harms both individuals and communities, making automatic detection essential for large-scale moderation. However, accurately detecting hate speech remains a difficult task. Part of the challenge lies in subjectivity: what one person flags as hate speech, another may see as benign. Traditional annotation agreement metrics, such as Cohen’s k, oversimplify this disagreement, treating it as an error rather than meaningful diversity. Meanwhile, Large Language Models (LLMs) promise scalable annotation, but prior studies demonstrate that they cannot fully replace human judgement, especially in subjective tasks. In this work, we reexamine LLM reliability using a subjectivity-aware framework, cross-Replication Reliability (xRR), revealing that even under fairer lens, LLMs still diverge from humans. Yet this limitation opens an opportunity: we find that LLM-generated annotations can reliably reflect performance trends across classification models, correlating with human evaluations. We test this by examining whether LLM-generated annotations preserve the relative ordering of model performance derived from human evaluation (i.e. whether models ranked as more reliable by human annotators preserve the same order when evaluated with LLM-generated labels). Our results show that, although LLMs differ from humans at the instance level, they reproduce similar ranking and classification patterns, suggesting their potential as proxy evaluators. While not a substitute for human annotators, they might serve as a scalable proxy for evaluation in subjective NLP tasks.}
}
```

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

The authors thank the funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them. The authors thank the financial support supplied by the grant PID2022-137061OB-C21 funded by MI-CIU/AEI/10.13039/501100011033 and by “ERDF/EU”. The authors also thank the funding supplied by the Consellería de Cultura, Educación, Formación Profesional e Universidades (accreditations ED431G 2023/01 and ED431C 2025/49) and the European Regional Development Fund, which acknowledges the CITIC, as a center accredited for excellence within the Galician University System and a member of the CIGUS Network, receives subsidies from the Department of Education, Science, Universities, and Vocational Training of the Xunta de Galicia. Additionally, it is co-financed by the EU through the FEDER Galicia 2021-27 operational program (Ref. ED431G 2023/01).


## 📜 License

This project is licensed under the Apache License 2.0 – see the LICENSE
 file for details.


## 📬 Contact

For further questions, inquiries, or discussions related to this project, please feel free to reach out via email:

- **Email:** [paloma.piot@udc.es](mailto:paloma.piot@udc.es)
