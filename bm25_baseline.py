from datasets import load_dataset
from rank_bm25 import BM25Okapi
import statistics
import math


BASE = "primer-ai/retrieval-response"

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(
    BASE,
    data_files="samples_texts.json",
    split="train"
)

print("Total queries:", len(dataset))


# ------------------------------------------------------------
# Evaluation settings
# ------------------------------------------------------------

K_VALUES = [1, 3, 5, 10]

all_precision = {k: [] for k in K_VALUES}
all_recall = {k: [] for k in K_VALUES}
all_ndcg = {k: [] for k in K_VALUES}


# ------------------------------------------------------------
# Evaluate queries
# ------------------------------------------------------------

print("\nRunning BM25 retrieval...")

for i, row in enumerate(dataset):

    query = row["q"]
    positives = row["p"]
    negatives = row["n"]

    # All candidate passages
    documents = positives + negatives

    # Tokenize documents
    tokenized_documents = [
        doc.lower().split()
        for doc in documents
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_documents)

    # Tokenize query
    tokenized_query = query.lower().split()

    # Retrieve scores
    scores = bm25.get_scores(tokenized_query)

    # Rank documents by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda x: scores[x],
        reverse=True
    )

    # Number of relevant documents
    relevant_count = len(positives)

    # Evaluate each K
    for k in K_VALUES:

        top_k = ranked_indices[:k]

        # Positive passages are indices 0 ... len(positives)-1
        relevant_retrieved = sum(
            1
            for idx in top_k
            if idx < relevant_count
        )

        precision = relevant_retrieved / k

        recall = (
            relevant_retrieved / relevant_count
            if relevant_count > 0
            else 0
        )
        # Calculate nDCG@K
        dcg = 0

        for rank, idx in enumerate(top_k, start=1):
            if idx < relevant_count:
                dcg += 1 / math.log2(rank + 1)

        ideal_k = min(k, relevant_count)

        idcg = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, ideal_k + 1)
        )

        ndcg = dcg / idcg if idcg > 0 else 0

        all_ndcg[k].append(ndcg)

        all_precision[k].append(precision)
        all_recall[k].append(recall)

    # Progress
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(dataset)} queries")


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BM25 RETRIEVAL RESULTS")
print("=" * 70)

# Store results so they can be used later for comparison and graphs
results = []

print("\nPrecision@K:")

for k in K_VALUES:
    value = statistics.mean(all_precision[k])
    line = f"Precision@{k}: {value:.4f}"
    print(line)
    results.append(line)


print("\nRecall@K:")

for k in K_VALUES:
    value = statistics.mean(all_recall[k])
    line = f"Recall@{k}: {value:.4f}"
    print(line)
    results.append(line)


print("\nF1@K:")

for k in K_VALUES:
    precision = statistics.mean(all_precision[k])
    recall = statistics.mean(all_recall[k])

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    line = f"F1@{k}: {f1:.4f}"
    print(line)
    results.append(line)


print("\nnDCG@K:")

for k in K_VALUES:
    value = statistics.mean(all_ndcg[k])
    line = f"nDCG@{k}: {value:.4f}"
    print(line)
    results.append(line)


print("\n" + "=" * 70)
print("BM25 BASELINE COMPLETE")
print("=" * 70)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

with open("results/bm25_results.txt", "w") as f:

    f.write("BM25 RETRIEVAL RESULTS\n")
    f.write("=" * 50 + "\n\n")

    f.write("Precision@K:\n")
    for k in K_VALUES:
        value = statistics.mean(all_precision[k])
        f.write(f"Precision@{k}: {value:.4f}\n")

    f.write("\nRecall@K:\n")
    for k in K_VALUES:
        value = statistics.mean(all_recall[k])
        f.write(f"Recall@{k}: {value:.4f}\n")

    f.write("\nF1@K:\n")
    for k in K_VALUES:
        precision = statistics.mean(all_precision[k])
        recall = statistics.mean(all_recall[k])

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        f.write(f"F1@{k}: {f1:.4f}\n")

    f.write("\nnDCG@K:\n")
    for k in K_VALUES:
        value = statistics.mean(all_ndcg[k])
        f.write(f"nDCG@{k}: {value:.4f}\n")

print("\nResults saved to: results/bm25_results.txt")