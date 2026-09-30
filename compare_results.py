# ------------------------------------------------------------
# Compare BM25 and Semantic Retrieval
# ------------------------------------------------------------

K_VALUES = [1, 3, 5, 10]

# ------------------------------------------------------------
# BM25 results
# ------------------------------------------------------------

bm25 = {
    "precision": {
        1: 0.5717,
        3: 0.3848,
        5: 0.3015,
        10: 0.2159
    },

    "recall": {
        1: 0.1957,
        3: 0.3754,
        5: 0.4698,
        10: 0.6241
    },

    "f1": {
        1: 0.2916,
        3: 0.3801,
        5: 0.3673,
        10: 0.3209
    },

    "ndcg": {
        1: 0.5717,
        3: 0.4788,
        5: 0.5003,
        10: 0.5649
    }
}


# ------------------------------------------------------------
# Semantic retrieval results
# ------------------------------------------------------------

semantic = {
    "precision": {
        1: 0.6878,
        3: 0.4743,
        5: 0.3628,
        10: 0.2455
    },

    "recall": {
        1: 0.2382,
        3: 0.4655,
        5: 0.5652,
        10: 0.7035
    },

    "f1": {
        1: 0.3538,
        3: 0.4699,
        5: 0.4419,
        10: 0.3640
    },

    "ndcg": {
        1: 0.6878,
        3: 0.5851,
        5: 0.6043,
        10: 0.6626
    }
}


# ------------------------------------------------------------
# Print comparison
# ------------------------------------------------------------

print("=" * 70)
print("BM25 vs SEMANTIC RETRIEVAL")
print("=" * 70)


for metric in ["precision", "recall", "f1", "ndcg"]:

    print(f"\n{metric.upper()}")

    for k in K_VALUES:

        bm25_value = bm25[metric][k]
        semantic_value = semantic[metric][k]

        improvement = semantic_value - bm25_value

        print(
            f"@{k}: "
            f"BM25={bm25_value:.4f} | "
            f"Semantic={semantic_value:.4f} | "
            f"Difference={improvement:+.4f}"
        )


# ------------------------------------------------------------
# F1 improvement
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("F1 IMPROVEMENT")
print("=" * 70)

for k in K_VALUES:

    improvement = semantic["f1"][k] - bm25["f1"][k]

    print(f"F1@{k}: {improvement:+.4f}")


# ------------------------------------------------------------
# nDCG improvement
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("nDCG IMPROVEMENT")
print("=" * 70)

for k in K_VALUES:

    improvement = semantic["ndcg"][k] - bm25["ndcg"][k]

    print(f"nDCG@{k}: {improvement:+.4f}")


# ------------------------------------------------------------
# Overall conclusion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(
    "\nSemantic retrieval achieved higher Precision, Recall, "
    "F1, and nDCG than BM25 at all evaluated K values."
)

largest_f1_k = max(
    K_VALUES,
    key=lambda k: semantic["f1"][k] - bm25["f1"][k]
)

largest_f1_improvement = (
    semantic["f1"][largest_f1_k]
    - bm25["f1"][largest_f1_k]
)

print(
    f"Largest F1 improvement: "
    f"+{largest_f1_improvement:.4f} at K={largest_f1_k}"
)