"""
Scrapes Google News for news articles related to a specific stock ticker (e.g., BTC-USD)
and analyzes the sentiment of each using the ProsusAI FinBERT model.

It uses:
- pygooglenews: to fetch news article links and titles from Google News.
- newspaper3k: to download and parse the full content of each news article.
- sentiment_bert.py: to run sentiment analysis on the article text.

The pygooglenews library is available at: https://github.com/kotartemiy/pygooglenews#about
The newspaper3k library is available at: https://github.com/codelucas/newspaper
The ProsusAI/finbert model is available on Hugging Face: https://huggingface.co/ProsusAI/finbert

Overall prediction score = sum of weighted scores / sum of all confidence scores
overall confidence score = sum of all cofidence scores/ number of articles
"""

from sentiment_bert import predict_sentiment
from pygooglenews import GoogleNews
import time

# cryptocurrency search queries for Google News
BTCUSDT = 'Bitcoin OR BTC OR Crypto'
BNBUSDT = 'Binance OR BNB OR Crypto'
ETHUSDT = 'Ethereum OR ETH OR Crypto'
DOGEUSDT = 'Dogecoin OR DOGE OR Crypto'
SHIBUSDT = 'Shiba OR SHIB OR Crypto'

# Scrapes Google News for headlines
def scrape (crypto_coin):
    gn = GoogleNews()
    search_entries = gn.search(crypto_coin)
    article_list = []
    for entry in search_entries["entries"]:
        article_list.append(entry["title"])
        # Set a delay betwen processing articles to avoid errors in news servers.
        time.sleep(0.25)
    return article_list


# Returns the following:
#  Overall Score = (sum of weighted scores) / (sum of all confidence score)
#  Overall confidence = (sum of all confidence scores) / (number of articles)
def determine_sentiment(crypto_coin):
    # Get list of news headlines 
    article_list = scrape(crypto_coin)
    # Error handler for when no articles are found
    if not article_list:
        return 0.0, 0.0
    
    total_score = 0
    total_confidence = 0

    for entry in article_list:
        # Call FinBert model for sentiment prediction and confidence score
        prediction, confidence = predict_sentiment(entry)

        # Get simple average.
        # Negative:-1 , Neutral: 0, Positive: 1 
        total_score += prediction
        total_confidence += confidence

    # Get average score and confidence
    final_score = total_score/len(article_list)
    final_confidence = total_confidence/len(article_list)

    return final_score, final_confidence


# Script testing functionality
if __name__ == "__main__":
    final_score, final_confidence = determine_sentiment(BNBUSDT)
    print(f"Final Score: {final_score}")
    print(f"Final Confidence: {final_confidence}\n")


    final_score, final_confidence = determine_sentiment(BTCUSDT)
    print(f"Final Score: {final_score}")
    print(f"Final Confidence: {final_confidence}\n")


    final_score, final_confidence = determine_sentiment(SHIBUSDT)
    print(f"Final Score: {final_score}")
    print(f"Final Confidence: {final_confidence}\n")


    final_score, final_confidence = determine_sentiment(ETHUSDT)
    print(f"Final Score: {final_score}")
    print(f"Final Confidence: {final_confidence}\n")



