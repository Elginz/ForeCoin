# Copy for 13 Aug 2025 14:43 
------------------------------------------------
- Going to implement the  rfr quantile regression model. It is still RFR model for now.

- After that i will be running a new backtesting.py file to see if it can read and test the effectiveness of the models

- start writing report and submit it to cher by then. 

- When that all works. i will consider setting a retraining model. that model should retrain when backtesting.py shows failed result 

- Then i focus on UI: Dashoard needs to be implemented, and candlestick data is not properly shown


------------------------------------------------
Directory for each file 

# sentiment analysis 
- webscraper.py
- sentiment_bert.py (for webscraper.py)
- test_Bert_to_Vader.ipynb (Chose FinBert to Vader)
- test_finBERT_to_LSTMBert.ipynb (Chose FinBert to LSTM with BERT)

need to determine what are the hperparameters to determine the thresholds

# chronos T5 Model
- predict_chronos.py
- test_chronos.ipynb (to determine effectiveness of chronos model)

# RFR & KNN Model
- models.py *NOT COMPLETED*

# Data Collection
- live_data.py
- data_collect.py (helper function to collect data) *NOT COMPLETED*

# App Hosting 
- app.py
- dashboard.html *NOT COMPLETED*
- dashboard_updater.js *NOT COMPLETED*
- stable.html
- stable_updater.js *NOT COMPLETED*
- volatile.html *NOT COMPLETED*
- volatile_updater.js *NOT COMPLETED*
- Sidebar.html *NOT COMPLETED*
- Footer.html *NOT COMPLETED*
- Settings-box.html *NOT COMPLETED*

------------------------------------------------
# LINKS AND REFERENCES 

*WEB DEVELOPMENT*

This is the theme of my websites
https://themesberg.com/product/flask/volt-admin-dashboard-template

*CHRONOS PREDICTION*

This is a model used to determine prices
https://www.linkedin.com/posts/raghu-nandan-82a23b283_bitcoin-price-trend-analysis-using-amazon-activity-7221110144481337346-fY1w/


*SENTIMENT ANALYSIS*

This is the dataset used to train BERT
https://www.kaggle.com/code/khotijahs1/nlp-financial-news-sentiment-analysis/input

https://huggingface.co/ProsusAI/finbert

This is used to scrape the internet
https://github.com/kotartemiy/pygooglenews#about


<!-- ADDITIONAL OPTIONS TO LOOK AT -->

FinGPT for reinforcement learning and sentiment analysis for stock trading
https://github.com/AI4Finance-Foundation/FinGPT
https://github.com/AI4Finance-Foundation/FinGPT?tab=readme-ov-file


<!-- Modules to produce -->

Data Collection: Gathers all Market information from crypto exchange.

Data Preparation: Cleans up the raw data, checks it for errors, and turns it into a useful format with important features for our AI.

AI Models: holds two different AI models, each specifically designed for different types of cryptocurrencies.

Model Training: This part takes what the AI models say and turns it into simple trading advice, like "BUY," "SELL," or "HOLD

Backtesting: Test the trading strategies using past market data.

User Interface and Deployment: The user-facing dashboard designed for all users.
