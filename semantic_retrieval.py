from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import statistics
import math
import os


BASE = "primer-ai/retrieval-response"

K_VALUES = [1, 3, 5, 10]

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
# Load semantic model
# ------------------------------------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


# ------------------------------------------------------------
# Evaluation settings
# ------------------------------------------------------------

all_precision = {k: [] for k in K_VALUES}
all_recall = {k: [] for k in K_VALUES}
all_ndcg = {k: [] for k in K_VALUES}


# ------------------------------------------------------------
# Evaluate queries
# ------------------------------------------------------------

print("\nRunning semantic retrieval...")

for i, row in enumerate(dataset):

    query = row["q"]
    positives = row["p"]
    negatives = row["n"]

    documents = positives + negatives

    # Encode query and documents
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    document_embeddings = model.encode(
        documents,
        convert_to_tensor=True
    )

    # Calculate cosine similarity
    scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    # Rank documents
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda x: float(scores[x]),
        reverse=True
    )

    relevant_count = len(positives)

    # Evaluate each K
    for k in K_VALUES:

        top_k = ranked_indices[:k]

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

        # nDCG@K
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

        all_precision[k].append(precision)
        all_recall[k].append(recall)
        all_ndcg[k].append(ndcg)

    # Progress
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(dataset)} queries")


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/semantic_results.txt"

with open(output_file, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("SEMANTIC RETRIEVAL RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Precision@K:\n")

    for k in K_VALUES:
        value = statistics.mean(all_precision[k])
        line = f"Precision@{k}: {value:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\nRecall@K:\n")

    for k in K_VALUES:
        value = statistics.mean(all_recall[k])
        line = f"Recall@{k}: {value:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\nF1@K:\n")

    for k in K_VALUES:

        precision = statistics.mean(all_precision[k])
        recall = statistics.mean(all_recall[k])

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        line = f"F1@{k}: {f1:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\nnDCG@K:\n")

    for k in K_VALUES:
        value = statistics.mean(all_ndcg[k])
        line = f"nDCG@{k}: {value:.4f}"
        print(line)
        f.write(line + "\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("SEMANTIC RETRIEVAL COMPLETE\n")
    f.write("=" * 70 + "\n")


print("\nResults saved to:", output_file)
print("\n" + "=" * 70)
print("SEMANTIC RETRIEVAL COMPLETE")
print("=" * 70)