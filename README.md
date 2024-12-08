
# Stock Movement Prediction

This repository contains the code and resources for predicting stock movements based on social media sentiment analysis from Reddit. The project involves data scraping, preprocessing, sentiment analysis, and training a logistic regression model to predict stock movement.

---

## Table of Contents
1. [Setup](#setup)
2. [Data Scraping](#data-scraping)
3. [Data Preprocessing](#data-preprocessing)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Running the Code](#running-the-code)
7. [Acknowledgements](#acknowledgements)

---

## Setup

### Prerequisites
To run this project, ensure you have the following installed:
- Python 3.x
- pip
- Virtualenv (recommended for managing dependencies)
- Reddit API Credentials (client ID, client secret, user agent)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-movement-prediction.git
   cd stock-movement-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv stock-env
   source stock-env/bin/activate  # On Windows: stock-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Scraping

The script `stockscraper.py` uses PRAW to scrape data from specified subreddits. It collects information such as titles, selftext, scores, and post timestamps.

### Running the Script
1. Replace the Reddit API credentials in `stockscraper.py` with your own.
2. Run the script:
   ```bash
   python stockscraper.py
   ```
3. Output: A CSV file named `reddit_stock_data.csv` containing the scraped data.

---

## Data Preprocessing

The script `cleaned_data.py` processes the raw Reddit data by:
- Filling missing values.
- Combining the title and selftext into a single column.
- Cleaning the text by removing URLs and special characters.

### Running the Script
1. Ensure `reddit_stock_data.csv` exists in the directory.
2. Run the script:
   ```bash
   python cleaned_data.py
   ```
3. Output: A CSV file named `cleaned_reddit_stock_data.csv`.

---

## Sentiment Analysis

The script `sentiment_analysis.py` performs sentiment analysis using TextBlob. It calculates:
- Sentiment polarity.
- Word frequency for "stock."
- Post creation hour and other features for model training.

### Running the Script
1. Ensure `cleaned_reddit_stock_data.csv` exists in the directory.
2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```
3. Output: The following CSV files will be generated for model training:
   - `X_train.csv`
   - `X_test.csv`
   - `y_train.csv`
   - `y_test.csv`

---

## Model Training and Evaluation

The script `train_model.py` trains a logistic regression model and evaluates its performance.

### Running the Script
1. Ensure the preprocessed files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`) exist.
2. Run the script:
   ```bash
   python train_model.py
   ```
3. Output:
   - Model accuracy and evaluation metrics.
   - Trained model saved as `stock_movement_model.pkl`.

---

## Running the Code

To execute the entire pipeline:
1. Run data scraping:
   ```bash
   python stockscraper.py
   ```
2. Run data preprocessing:
   ```bash
   python cleaned_data.py
   ```
3. Perform sentiment analysis:
   ```bash
   python sentiment_analysis.py
   ```
4. Train and evaluate the model:
   ```bash
   python train_model.py
   ```

---

## Acknowledgements

- **Reddit API** for providing access to social media data.
- **TextBlob** for sentiment analysis.
- **Pandas** for data manipulation.
- **scikit-learn** for model training and evaluation.

---



