import praw
import pandas as pd

# Step 1: Reddit API Credentials
client_id = "reddit id"
client_secret = "secret code"
user_agent = "user agent"

# Step 2: Initialize PRAW
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Step 3: Define Subreddits to Scrape
subreddits = ["WallStreetBets", "StockMarket", "Investing"]

# Step 4: Scrape Data
def scrape_reddit(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.new(limit=limit):  # Scrape the newest posts
        posts.append({
            "title": post.title,
            "selftext": post.selftext,
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": post.created_utc,
            "url": post.url
        })

    return posts

# Step 5: Collect Data from All Subreddits
data = []
for sub in subreddits:
    print(f"Scraping {sub}...")
    data.extend(scrape_reddit(sub, limit=100))

# Step 6: Save Data to CSV
df = pd.DataFrame(data)
df.to_csv("reddit_stock_data.csv", index=False)
print("Data saved to reddit_stock_data.csv!")
