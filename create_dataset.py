import pandas as pd


def load_file(filepath):
    reviews = []
    labels = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            review, label = line.rsplit("\t", 1)

            reviews.append(review)
            labels.append(int(label))

    return pd.DataFrame({
        "verified_reviews": reviews,
        "feedback": labels
    })


amazon_df = load_file("data/amazon_cells_labelled.txt")
imdb_df = load_file("data/imdb_labelled.txt")
yelp_df = load_file("data/yelp_labelled.txt")

combined_df = pd.concat(
    [amazon_df, imdb_df, yelp_df],
    ignore_index=True
)

combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

combined_df.to_csv(
    "data/sentiment_dataset.csv",
    index=False
)

print("Dataset Created Successfully")
print()
print("Shape:", combined_df.shape)
print()
print(combined_df["feedback"].value_counts())