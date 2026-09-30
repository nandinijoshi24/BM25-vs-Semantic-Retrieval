# Data

The experiments used a retrieval dataset containing 6,112 text samples.

The full raw dataset is not included in this public repository. Place the permitted dataset file(s) here before running the experiments.

The dataset used in the project contained fields for identifiers, questions, passages/negative passages and candidate response-related text.

## Project Structure

```text
BM25-vs-Semantic-Retrieval/
├── bm25_baseline.py
├── semantic_retrieval.py
├── compare_results.py
├── dataset_stats.py
├── inspect_dataset.py
├── plot_results.py
├── bm25_results.txt
├── semantic_results.txt
├── results_and_findings.txt
├── precision_comparison.csv
├── semantic_ndcg.csv
├── precision_comparison.png
├── recall_comparison.png
├── f1_comparison.png
├── ndcg_comparison.png
├── requirements.txt
├── README.md
└── .gitignore
