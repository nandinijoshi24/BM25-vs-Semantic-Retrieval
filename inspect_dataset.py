from datasets import load_dataset

BASE = "primer-ai/retrieval-response"

files = [
    "samples_texts.json",
    "samples_ranked.json",
    "samples_graded.json",
    "samples_human.json"
]

for file in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    try:
        dataset = load_dataset(
            BASE,
            data_files=file,
            split="train"
        )

        print("Rows:", len(dataset))
        print("Columns:", dataset.column_names)
        print("\nFirst record:")
        print(dataset[0])

    except Exception as e:
        print("ERROR:", e)