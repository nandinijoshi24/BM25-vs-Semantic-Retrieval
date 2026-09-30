import matplotlib.pyplot as plt

K_VALUES = [1, 3, 5, 10]

bm25 = {
    "precision": [0.5717, 0.3848, 0.3015, 0.2159],
    "recall": [0.1957, 0.3754, 0.4698, 0.6241],
    "f1": [0.2916, 0.3801, 0.3673, 0.3209],
    "ndcg": [0.5717, 0.4788, 0.5003, 0.5649]
}

semantic = {
    "precision": [0.6878, 0.4743, 0.3628, 0.2455],
    "recall": [0.2382, 0.4655, 0.5652, 0.7035],
    "f1": [0.3538, 0.4699, 0.4419, 0.3640],
    "ndcg": [0.6878, 0.5851, 0.6043, 0.6626]
}


def create_graph(metric, title, filename):

    plt.figure(figsize=(8, 5))

    plt.plot(
        K_VALUES,
        bm25[metric],
        marker="o",
        label="BM25"
    )

    plt.plot(
        K_VALUES,
        semantic[metric],
        marker="o",
        label="Semantic Retrieval"
    )

    plt.xlabel("K")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.xticks(K_VALUES)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"results/{filename}", dpi=300)
    plt.show()


create_graph(
    "precision",
    "Precision@K: BM25 vs Semantic Retrieval",
    "precision_comparison.png"
)

create_graph(
    "recall",
    "Recall@K: BM25 vs Semantic Retrieval",
    "recall_comparison.png"
)

create_graph(
    "f1",
    "F1@K: BM25 vs Semantic Retrieval",
    "f1_comparison.png"
)

create_graph(
    "ndcg",
    "nDCG@K: BM25 vs Semantic Retrieval",
    "ndcg_comparison.png"
)

print("\nAll graphs saved successfully!")