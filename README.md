# BM25 vs Semantic Retrieval

A comparative Information Retrieval project evaluating traditional lexical retrieval with BM25 against semantic retrieval using sentence embeddings.

## Project Overview

This project implements and compares two retrieval approaches:

1. **BM25** — a lexical retrieval method based on term matching and document relevance.
2. **Semantic Retrieval** — retrieves text using sentence embeddings and semantic similarity.

The retrieval results are evaluated using standard Information Retrieval metrics to study differences in retrieval quality.

## Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
┌─────────────────────┬─────────────────────────┐
│ BM25 Retrieval      │ Semantic Retrieval      │
│ Lexical Matching    │ Sentence Embeddings    │
└─────────────────────┴─────────────────────────┘
             ↓
        Ranking Results
             ↓
     Evaluation Metrics
             ↓
       Comparative Analysis
```

## Dataset

The project uses a retrieval dataset containing **6,112 text samples**.

The working dataset includes fields used for questions, passages, negative examples and candidate responses.

> The full raw dataset is intentionally not included in this public repository. Add or download the dataset separately according to its original source and redistribution terms.

## Evaluation Metrics

The project evaluates retrieval quality using:

- **Precision@K**
- **Recall@K**
- **F1**
- **nDCG (Normalized Discounted Cumulative Gain)**

The evaluation is performed at multiple retrieval depths such as K = 1, 3, 5 and 10.

## Results

### Precision Comparison

| K | BM25 Precision | Semantic Precision |
|---:|---:|---:|
| 1 | 0.5717 | 0.6878 |
| 3 | 0.3848 | 0.4743 |
| 5 | 0.3015 | 0.3628 |
| 10 | 0.2159 | 0.2455 |

### Semantic Retrieval nDCG

| K | Semantic nDCG |
|---:|---:|
| 1 | 0.6878 |
| 3 | 0.5851 |
| 5 | 0.6043 |
| 10 | 0.6626 |

These values are the recorded results from the project experiments. The repository does not claim that one retrieval approach is universally better; retrieval quality depends on the metric, dataset and evaluation setting.

## Project Structure

```text
BM25-vs-Semantic-Retrieval/
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── results/
│   ├── precision_comparison.csv
│   ├── semantic_ndcg.csv
│   └── graphs/
├── src/
│   └── README.md
├── .gitignore
├── README.md
└── requirements.txt
```

## Technologies

- Python
- BM25
- Sentence Transformers
- Semantic Embeddings
- Information Retrieval
- NumPy
- Pandas
- Scikit-learn

## Key Learning Areas

- Lexical vs semantic retrieval
- Ranking and retrieval evaluation
- Precision, Recall and F1
- nDCG
- Sentence embeddings
- Comparative result analysis

## How to Run

The source-code files used for the experiments can be placed inside `src/`, and notebooks can be placed inside `notebooks/`.

Install the required packages:

```bash
pip install -r requirements.txt
```

Then place the permitted dataset in the `data/` directory and run the relevant retrieval and evaluation scripts/notebooks.

## Note

This repository is intended to document the implementation and evaluation work for the BM25 vs Semantic Retrieval project. Dataset redistribution should follow the terms of the original dataset source.
