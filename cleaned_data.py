import pandas as pd
import re

# Load the data
file_path = "reddit_stock_data.csv"
df = pd.read_csv(file_path)

# Fill missing selftext with placeholder
df["selftext"] = df["selftext"].fillna("No content")

# Combine 'title' and 'selftext'
df["content"] = df["title"] + " " + df["selftext"]

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["content"] = df["content"].apply(clean_text)

# Add subreddit column if needed
# df["subreddit"] = "WallStreetBets"

# Select relevant columns (adjust based on actual columns in your data)
df = df[["content", "score", "num_comments", "created_utc"]]

# Save the cleaned data
df.to_csv("cleaned_reddit_stock_data.csv", index=False)
print("Cleaned data saved to cleaned_reddit_stock_data.csv!")

# Load and preview the cleaned data
cleaned_df = pd.read_csv("cleaned_reddit_stock_data.csv")
print("Cleaned Data Preview:")
print(cleaned_df.head())
