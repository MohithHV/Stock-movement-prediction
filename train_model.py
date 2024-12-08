import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Load the target variables from CSV files without headers
y_train_df = pd.read_csv('y_train.csv', header=None)
y_train = y_train_df[0]  # Adjust if column is not at index 0

y_test_df = pd.read_csv('y_test.csv', header=None)
y_test = y_test_df[0]  # Adjust if column is not at index 0

# Ensure that y_train matches the number of samples in X_train
if len(X_train) != len(y_train):
    raise ValueError(f"Mismatch in number of samples: X_train has {len(X_train)} samples, but y_train has {len(y_train)} samples.")

# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict stock movements on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model
joblib.dump(model, 'stock_movement_model.pkl')
print("Trained model saved as stock_movement_model.pkl!")
