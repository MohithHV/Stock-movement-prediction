import pandas as pd

# Load the cleaned Reddit stock data
cleaned_file_path = "cleaned_reddit_stock_data.csv"
df = pd.read_csv(cleaned_file_path)

# Display the first few rows
print("Cleaned Data Preview:")
print(df.head())

from textblob import TextBlob

# Function to perform sentiment analysis
def get_sentiment_polarity(text):
    """
    Returns the polarity of the text.
    Polarity ranges from -1 (negative) to 1 (positive).
    """
    return TextBlob(text).sentiment.polarity

# Apply sentiment analysis
df['sentiment_polarity'] = df['content'].apply(get_sentiment_polarity)

# Display a preview of the dataframe with sentiment scores
print("\nData with Sentiment Polarity:")
print(df[['content', 'sentiment_polarity']].head())

# Feature: Frequency of the word 'stock'
df['stock_mentions'] = df['content'].apply(lambda x: x.lower().count('stock'))

# Feature: Post time (hour of the day)
df['post_hour'] = pd.to_datetime(df['created_utc']).dt.hour

# Feature: Number of comments
df['num_comments'] = df['num_comments']

# Check the first few rows with extracted features
print("\nData with Extracted Features:")
print(df[['content', 'stock_mentions', 'post_hour', 'num_comments', 'sentiment_polarity']].head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assume we have a target column 'stock_movement' indicating if the stock moved up or down
# For demonstration purposes, let's create a dummy target column
# This should be replaced with actual stock movement data

# For example, a label 1 could indicate an increase, 0 a decrease
# Here we generate random labels for demonstration
import numpy as np
np.random.seed(0)
df['stock_movement'] = np.random.randint(0, 2, size=len(df))  # Random binary labels

# Select features and target
X = df[['sentiment_polarity', 'stock_mentions', 'post_hour', 'num_comments']]
y = df['stock_movement']  # Target variable

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# Output dataset sizes for verification
print("\nX_train preview:")
print(X_train.head())

print("\nX_test preview:")
print(X_test.head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
