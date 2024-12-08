import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the cleaned Reddit stock data
cleaned_file_path = "cleaned_reddit_stock_data.csv"
df = pd.read_csv(cleaned_file_path)

# Display a preview of the cleaned data
print("Cleaned Data Preview:")
print(df.head())

# Function to calculate sentiment polarity
def get_sentiment_polarity(text):
    """
    Returns the polarity of the text.
    Polarity ranges from -1 (negative) to 1 (positive).
    """
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
df['sentiment_polarity'] = df['content'].apply(get_sentiment_polarity)

# Feature: Count of the word 'stock'
df['stock_mentions'] = df['content'].apply(lambda x: x.lower().count('stock'))

# Feature: Hour of the day when the post was created
df['post_hour'] = pd.to_datetime(df['created_utc'], unit='s').dt.hour

# Feature: Number of comments on the post
df['num_comments'] = df['num_comments']

# Display the first few rows with extracted features
print("\nData with Extracted Features:")
print(df[['content', 'stock_mentions', 'post_hour', 'num_comments', 'sentiment_polarity']].head())

# Assume we have a target column 'stock_movement' indicating if the stock moved up or down
# For demonstration purposes, let's create a dummy target column
# This should be replaced with actual stock movement data
np.random.seed(0)
df['stock_movement'] = np.random.randint(0, 2, size=len(df))  # Random binary labels

# Select features and target variable for model training
X = df[['sentiment_polarity', 'stock_mentions', 'post_hour', 'num_comments']]
y = df['stock_movement']  # Target variable

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# Output dataset sizes for verification
print("\nX_train preview:")
print(X_train.head())

print("\nX_test preview:")
print(X_test.head())

# Save preprocessed data for later use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

# Convert y_train and y_test to DataFrame before saving
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

print("Preprocessed data saved!")


