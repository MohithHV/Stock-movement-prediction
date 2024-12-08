import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Load the target variables from CSV files and access the correct column names
y_train_df = pd.read_csv('y_train.csv')
y_train = y_train_df['movement']  # Adjust this column name if needed

y_test_df = pd.read_csv('y_test.csv')
y_test = y_test_df['movement']  # Adjust this column name if needed

# Check for class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(random_state=42)

# Hyperparameter tuning using Grid Search
params = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2']
}
grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Predict stock movements on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save the trained model
joblib.dump(best_model, 'stock_movement_model.pkl')
print("Trained model saved as stock_movement_model.pkl!")
