# ForeCoin 

This is the implementation of project 4.2: Financial Advisor Bot. It creates a bot that analyses financial data of crypto currencies
in order to make recommendations for a dynamic investment strategy. 

The bot currently works with 5 cryptocurrencies

Stable Coins: BTCUSDT, ETHUSDT, BNBUSDT
Volatile Coins: DOGEUSDT, SHIBUSDT
______
<h3 align="center">🖥️ How to run 🏃</h3>

Run the requirements.txt file 
- ```pip install -r requirements.txt```

then run the following 

- ```pip install --upgrade --force-reinstall "feedparser>=6.0.10"```

(As ```pygooglenews``` strictly requires ```feedparser``` to be less than version 6.0.0, we would need to forcibly upgrade it)

______
<h3 align="center">🗃️System Architecture🗃️</h3>

*insert image of system architecture here*

*Explain briefly on the system architecture*


```
ForeCoin/
│
├── apps/
│   ├── static/
│   │   └── assets/
│   │        ├── package.json/
│   |        ├── css/
│   |        ├── img/
│   |        ├── vendor/
│   |        └── js/
│   |            ├── dashboard_updater.js
│   |            ├── index_search.js
│   |            ├── stable_updater.js
│   |            ├── volatile_updater.js
│   |            └── volt.js
|   |
│   └── templates/
│       └── home/
│           ├── index.html
│           ├── dashboard.html
│           ├── stable.html
│           └── volatile.html
│       └── includes/
│           ├── footer.html
│           ├── navigation.html
│           ├── scripts.html
│           └── sidebar.html
│       └── layouts/
│           ├── base-fullscreen.html
│           └── base.html
|
├── historic_data/
│   ├── sentiment (can remove)/
│   ├── stable/
│   ├── volatile/
│   └── *all_coins_chronos_pred.csv*
│
├── trained_models/
│   ├── BNBUSDT_knn_supertrend_model.pkl
│   ├── BTCUSDT_knn_supertrend_model.pkl
│   ├── ETHUSDT_knn_supertrend_model.pkl
│   ├── DOGEUSDT_lgbm_quantile_model.pkl
│   ├── SHIBUSDT_lgbm_quantile_model.pkl
│   └── *all_coins_chronos_pred.csv*
|
├── trained_models/
│   ├── backtest_log.json
│   ├── BNBUSDT_predictions.json
│   ├── BTCUSDT_predictions.json
│   ├── ETHUSDT_predictions.json
│   ├── SHIBUSDT_predictions.json
│   └── dogeUSDT_predictions.json
|
├── app.py
├── backtesting.py
├── data_collect.py
├── models.py
├── predict_chronos.py
├── sentiment_bert.py
├── webscraper.py
├── readme.md
└── requirements.txt
```
______

<h4 align="center">AI models used</h4>

**KNN With Supertrend** 

add words here

**LGBM with Quantile Regression** 

add words here

**Chronos T5** 

add words here

**FinBERT for Sentiment Analysis** 

add words here

______
**Flask**
add words here

