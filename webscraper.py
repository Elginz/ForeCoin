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
        # Set a delay between processing articles to avoid errors in news servers.
        time.sleep(0.25)
    return article_list


# Returns the following:
#  Overall Score = (sum of weighted scores) / (sum of all confidence scores)
#  Overall confidence = (sum of all confidence scores) / (number of articles)
def determine_sentiment(crypto_coin):
    # Get list of news headlines 
    article_list = scrape(crypto_coin)

    # Error handler for when no articles are found
    if not article_list:
        return 0.0, 0.0  

    total_weighted_score = 0
    total_confidence = 0
    
    # Loop through the article title.
    for entry in article_list:
        # Call FinBert model for sentiment prediction and confidence score
        prediction, confidence = predict_sentiment(entry)
        
        # Multiply the prediction by its confidence to get a weighted score.
        total_weighted_score += prediction * confidence
        
        # Track the sum of all confidence scores 
        total_confidence += confidence
        
    # Normalize the score between -1.0 to 1.0.
    if total_confidence > 0:
        final_score = total_weighted_score / total_confidence
    else:
        # Error handler if divided by zero
        final_score = 0.0 

    # The final confidence is the average confidence across all articles.
    final_confidence = total_confidence / len(article_list)

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



