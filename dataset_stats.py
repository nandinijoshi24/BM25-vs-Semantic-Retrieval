from datasets import load_dataset
import statistics

BASE = "primer-ai/retrieval-response"


def load_file(filename):
    return load_dataset(
        BASE,
        data_files=filename,
        split="train"
    )


# ============================================================
# 1. SAMPLES_TEXTS
# ============================================================

print("=" * 70)
print("1. SAMPLES_TEXTS")
print("=" * 70)

texts = load_file("samples_texts.json")

print("Total records:", len(texts))
print("Columns:", texts.column_names)

# Query statistics
query_lengths = [
    len(str(q).split())
    for q in texts["q"]
    if q is not None
]

print("\nQuery statistics:")
print("Average query length:",
      round(statistics.mean(query_lengths), 2), "words")
print("Minimum query length:", min(query_lengths), "words")
print("Maximum query length:", max(query_lengths), "words")

# Positive / negative passage statistics
positive_counts = [
    len(p) if p is not None else 0
    for p in texts["p"]
]

negative_counts = [
    len(n) if n is not None else 0
    for n in texts["n"]
]

print("\nPassage statistics:")
print(
    "Average positive passages:",
    round(statistics.mean(positive_counts), 2)
)
print(
    "Average negative passages:",
    round(statistics.mean(negative_counts), 2)
)

print(
    "Minimum positive passages:",
    min(positive_counts)
)
print(
    "Maximum positive passages:",
    max(positive_counts)
)

print(
    "Minimum negative passages:",
    min(negative_counts)
)
print(
    "Maximum negative passages:",
    max(negative_counts)
)


# ============================================================
# 2. SAMPLES_RANKED
# ============================================================

print("\n" + "=" * 70)
print("2. SAMPLES_RANKED")
print("=" * 70)

ranked = load_file("samples_ranked.json")

print("Total records:", len(ranked))
print("Columns:", ranked.column_names)

print("\nEvaluation settings:")
print("Unique E values:", set(ranked["E"]))

print("\nExample K values:")
print(ranked[0]["K"])

print("\nExample Precision values:")
print(ranked[0]["P"])

print("\nExample Recall values:")
print(ranked[0]["R"])


# ============================================================
# 3. SAMPLES_GRADED
# ============================================================

print("\n" + "=" * 70)
print("3. SAMPLES_GRADED")
print("=" * 70)

graded = load_file("samples_graded.json")

print("Total records:", len(graded))
print("Columns:", graded.column_names)

grades = [
    g for g in graded["grade"]
    if g is not None
]

print("\nGrade statistics:")
print("Minimum grade:", min(grades))
print("Maximum grade:", max(grades))
print("Average grade:", round(statistics.mean(grades), 3))

# Grade distribution
print("\nGrade distribution:")

grade_counts = {}

for grade in grades:
    grade_counts[grade] = grade_counts.get(grade, 0) + 1

for grade in sorted(grade_counts):
    print(
        f"Grade {grade}: {grade_counts[grade]}"
    )


# ============================================================
# 4. SAMPLES_HUMAN
# ============================================================

print("\n" + "=" * 70)
print("4. SAMPLES_HUMAN")
print("=" * 70)

human = load_file("samples_human.json")

print("Total records:", len(human))
print("Columns:", human.column_names)

print("\nHuman evaluation statistics:")

for column in [
    "grade_human_1",
    "grade_human_2",
    "grade_human_3"
]:
    values = [
        x for x in human[column]
        if x is not None
    ]

    print(
        column,
        "average:",
        round(statistics.mean(values), 3)
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("DATASET STATISTICS COMPLETE")
print("=" * 70)