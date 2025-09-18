"""
This is the main app.py file

This file serves to continunously collect data on the respective assets.

It uses the data_collect.py helper functions
"""
# Main libraries
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import os
import time
import math
import json
import glob

# ML and data libraries
import joblib 
import pandas as pd
import pandas_ta as ta
import torch 

# Scheduling libaries
from datetime import datetime, timedelta
import websocket
import schedule

#  Import local python files
import data_collect
from predict_chronos import forecast_next_hour
from webscraper import determine_sentiment, BTCUSDT as BTC_QUERY, ETHUSDT as ETH_QUERY, BNBUSDT as BNB_QUERY, DOGEUSDT as DOGE_QUERY, SHIBUSDT as SHIB_QUERY

# --- App Configuration and Global Variables ---
# Paths for Flask web templates and statistic files
template_folder = os.path.join(os.getcwd(), 'apps', 'templates')
static_folder = os.path.join(os.getcwd(), 'apps', 'static')

# Main flask application instances
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.config['SECRET_KEY'] = 'secret_key_123!' 

# SocketIO instance for front end
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Asset lists and constants ---
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
ALL_ASSETS = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS

# --- Dictionary for display  ---
ASSET_NAMES = {
    'BTCUSDT': 'Bitcoin',
    'ETHUSDT': 'Ethereum',
    'BNBUSDT': 'Binance',
    'DOGEUSDT': 'Dogecoin',
    'SHIBUSDT': 'Shiba Inu'
}

# --- Map asset symbols to search queries ---
ASSET_QUERIES = { 'BTCUSDT': BTC_QUERY, 
                  'ETHUSDT': ETH_QUERY, 
                  'BNBUSDT': BNB_QUERY, 
                  'DOGEUSDT': DOGE_QUERY, 
                  'SHIBUSDT': SHIB_QUERY }

# Global settings
MAX_HISTORICAL_POINTS = 200 
BASE_DATA_FOLDER = "historic_data"

# --- Caches for storing live data ---
MODELS_DICT = {}
CHRONOS_MODELS = {} 
SENTIMENT_CACHE = {symbol: {} for symbol in ALL_ASSETS}
LAST_SENTIMENT_UPDATE = {symbol: None for symbol in ALL_ASSETS}
SENTIMENT_LOCK = threading.Lock()
HISTORICAL_DATA_CACHE = {symbol: [] for symbol in ALL_ASSETS}
LATEST_DATA_CACHE = {}

# --- Helper Functions ---

# Finds data CSV file for certain assets
def find_asset_data_file(symbol):
    if symbol in STABLE_ASSETS:
        subfolder = 'stable'
    else:
        subfolder = 'volatile'

    simple_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_data.csv")
    if os.path.exists(simple_path):
        return simple_path
    # Search for file using name, returning most recent
    search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# Format ISO timestamp into readable format for UI
def format_iso_timestamp(iso_string):
    if not iso_string or 'T' not in str(iso_string): 
        return 'N/A'
    try:
        dt_object = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt_object.strftime('%Y-%m-%d (%H:%M:%S UTC)')
    except (ValueError, TypeError): return 'N/A'

# Removes NAaN and infinite values 
def clean_nan_values(obj):
    if isinstance(obj, dict): return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_nan_values(i) for i in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

# Loads most recent data from CSV files into memory caches.
def populate_initial_caches(symbols):
    for symbol in symbols:
        # populate if the symbols cache is empty
        if not HISTORICAL_DATA_CACHE[symbol]:
            try:
                path = find_asset_data_file(symbol)
                if path and os.path.exists(path):
                    # Read CSV and get last data points
                    df = pd.read_csv(path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df_recent = df.tail(MAX_HISTORICAL_POINTS)
                    # Fill historical cache for chart
                    for _, row in df_recent.iterrows():
                        HISTORICAL_DATA_CACHE[symbol].append({
                            "time": row['timestamp'].timestamp(),
                            "open": row['open'], "high": row['high'],
                            "low": row['low'], "close": row['close'],
                            "volume": row['volume']
                        })
                    # Fill latest data cache
                    if not LATEST_DATA_CACHE.get(symbol) and not df_recent.empty:
                        last_row = df_recent.iloc[-1]
                        last_kline = { "time": last_row['timestamp'].timestamp(), 
                                      "open": last_row['open'], "high": last_row['high'], 
                                      "low": last_row['low'], "close": last_row['close'] }
                        
                        LATEST_DATA_CACHE[symbol] = { "symbol": symbol, "kline": last_kline, 
                                                     "prediction": {}, "sentiment": {}, 
                                                     "event_time_iso": last_row['timestamp'].isoformat() }
            except Exception as e:
                print(f"Error populating caches for {symbol}: {e}")

# --- Model Loading ---
# Loads the KNN and LGBM model 
def load_all_models(symbols):
    global MODELS_DICT
    models_dir = "trained_models"
    for symbol in symbols:
        # KNN for stable assets
        if symbol in STABLE_ASSETS:
            model_type = "KNN Supertrend"
            model_filename = f"{symbol}_knn_supertrend_model.pkl"
        else:
        # LGBM for volatile assets
            model_type = "LGBM Quantile"
            model_filename = f"{symbol}_lgbm_quantile_model.pkl"

        model_path = os.path.join(models_dir, model_filename)
        print(f"Attempting to load {model_type} model for {symbol}...")
        try:
            if os.path.exists(model_path):
                # Load model and store it in cache
                MODELS_DICT[symbol] = joblib.load(model_path)
                print(f"--> SUCCESS: {symbol}'s {model_type} model loaded.")
            else:
                # Error handler if file not found
                print(f"--> WARNING: Model file not found at '{model_path}'")
                MODELS_DICT[symbol] = None
        except Exception as e:
            print(f"--> ERROR: Could not load model for {symbol}: {e}")
            MODELS_DICT[symbol] = None

# Loads pretrained chronos model
def load_chronos_models():
    global CHRONOS_MODELS
    try:
        from chronos import ChronosPipeline
        for symbol in ALL_ASSETS:
            path = find_asset_data_file(symbol)
            if not path:
                print(f"Chronos Error: No data file found for {symbol}. Skipping model load.")
                continue
            try:
                # Load historical data
                df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
                values = df["close"].values
                context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
                # Load pre trained model
                pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
                # Store pipeline, context and df
                CHRONOS_MODELS[symbol] = {"pipeline": pipeline, "context": context, "df": df}
                print(f"Successfully loaded Chronos model for {symbol}.")
            except Exception as e:
                print(f"Error loading Chronos model for {symbol}: {e}"); CHRONOS_MODELS[symbol] = None
    except ImportError:
        print("Chronos library not found. Skipping Chronos model loading.")

# --- Background Threads ---
# Fetch and update the market sentiment
def update_single_sentiment(symbol):
    with SENTIMENT_LOCK:
        query = ASSET_QUERIES.get(symbol)
        if not query:
            return
        
        print(f"--- Updating sentiment for {symbol}... ---")
        try:
            # Call webscraper for sentiment score and confidence socre
            final_score, final_confidence = determine_sentiment(query)
            
            # Convert numerical score to neutral/postive/negative
            label = "neutral"
            if final_score > 0.05:
                label = "positive"
            elif final_score < -0.05:
                label = "negative"
            
            # Sentiment data and cache update
            sentiment_data = {"label": label, "score": final_score, "confidence": final_confidence}
            SENTIMENT_CACHE[symbol] = sentiment_data
            LAST_SENTIMENT_UPDATE[symbol] = datetime.now()
            
            # Sentiment data for all connected web clients
            socketio.emit('sentiment_update', {'symbol': symbol, 'sentiment': sentiment_data})
            print(f"Sentiment for {symbol}: {label} (Score: {final_score:.2f}) - Pushed to clients.")
        except Exception as e:
            print(f"An error occurred in sentiment update for {symbol}: {e}")

#  Prediction Logic
def process_kline_and_predict(kline_message_data):
    # Get candle and symbol data from websocket message
    kline_info = kline_message_data['k']
    symbol = kline_info['s']
    
    # Refresh sentiment data if it is older than 15 minutes 
    last_update = LAST_SENTIMENT_UPDATE.get(symbol)
    if not last_update or (datetime.now() - last_update) > timedelta(minutes=15):
        if not SENTIMENT_LOCK.locked():
            #  Run sentiment update in background thread
             socketio.start_background_task(update_single_sentiment, symbol=symbol)

    # placeholders
    knn_pred_val = None
    lgbm_preds = {'low': None, 'median': None, 'high': None}
    chronos_pred_val = None
    
    # loaded pre-trained model
    model_package = MODELS_DICT.get(symbol)
    
    if model_package:
        try:
            # STABLE assets (KNN model)
            if symbol in STABLE_ASSETS and isinstance(model_package, dict):
                history_data = HISTORICAL_DATA_CACHE.get(symbol, [])
                # Only need enough history to compute features
                if len(history_data) < 20: 
                    print(f"Not enough historical data for {symbol} to calculate features.")
                else:
                    # Convert history and current candle into a dataframe
                    history_df = pd.DataFrame(history_data)
                    current_kline_data = {
                        "open": float(kline_info['o']), "high": float(kline_info['h']),
                        "low": float(kline_info['l']), "close": float(kline_info['c']),
                        "volume": float(kline_info['v'])
                    }
                    current_kline_df = pd.DataFrame([current_kline_data])
                    combined_df = pd.concat([history_df, current_kline_df], ignore_index=True)
                    # Supertrend indicator
                    combined_df.ta.supertrend(length=10, multiplier=3.0, append=True)
                    supertrend_col = next((col for col in combined_df.columns if col.startswith('SUPERT_')), None)
                    # Feature engineering
                    combined_df['price_change'] = combined_df['close'].pct_change()
                    combined_df['volatility'] = combined_df['close'].rolling(window=10).std()
                    combined_df['volume_ma'] = combined_df['volume'].rolling(window=5).mean()
                    
                    # Handle NaN values after feature creation
                    combined_df.ffill(inplace=True) 
                    combined_df.dropna(inplace=True)

                    if not combined_df.empty:
                        # Get most recent row as input features
                        latest_features = combined_df.tail(1)
                        # load scikit-learn pipeline and list of feature columns
                        pipeline = model_package.get('pipeline')
                        expected_features = model_package.get('feature_columns')
                        
                        if pipeline and expected_features:
                            # Ensure feature columns match what the model expects
                            features_for_prediction = latest_features[expected_features]
                            knn_pred_val = pipeline.predict(features_for_prediction)[0]

            # VOLATILE assets (LGBM Quantile Model) 
            elif symbol in HIGH_VOLATILITY_ASSETS and isinstance(model_package, dict):
                # Ensure features_dict is defined correctly
                features_dict = {
                    'open': float(kline_info['o']), 'high': float(kline_info['h']),
                    'low': float(kline_info['l']), 'close': float(kline_info['c']),
                    'volume': float(kline_info['v'])
                }
                feature_order = ['open', 'high', 'low', 'close', 'volume']
                features_df = pd.DataFrame([features_dict], columns=feature_order)
                
                # Predict using all quantiles
                for name, model in model_package.items():
                    lgbm_preds[name] = model.predict(features_df)[0]

        except Exception as e:
            print(f"--> UNEXPECTED ERROR during model prediction for {symbol}: {e}")

    # Chronos Model
    if symbol in CHRONOS_MODELS and CHRONOS_MODELS[symbol] is not None:
        try:
            model_info = CHRONOS_MODELS[symbol]
            # update the context with closing price
            new_price_tensor = torch.tensor([float(kline_info['c'])], dtype=model_info["context"].dtype)
            model_info["context"] = torch.cat([model_info["context"], new_price_tensor.unsqueeze(0)], dim=1)[:, -1024:]
            # Forecast next hour
            _, median, _, _ = forecast_next_hour(model_info["df"], model_info["df"]["close"].values, model_info["context"], model_info["pipeline"])
            chronos_pred_val = median[0]
        except Exception as e:
            print(f"ERROR making Chronos prediction for {symbol}: {e}")

    # Combine all data into a single dict
    live_update_data = {
        "symbol": symbol,
        "kline": {"time": kline_info['t'] / 1000, "open": float(kline_info['o']), "high": float(kline_info['h']), "low": float(kline_info['l']), "close": float(kline_info['c'])},
        "prediction": {
            "knn_supertrend_close": float(knn_pred_val) if knn_pred_val is not None else None,
            "lgbm_quantiles": {
                "low": float(lgbm_preds['low']) if lgbm_preds['low'] is not None else None,
                "median": float(lgbm_preds['median']) if lgbm_preds['median'] is not None else None,
                "high": float(lgbm_preds['high']) if lgbm_preds['high'] is not None else None,
            },
            "chronos_next_hour_close": float(chronos_pred_val) if chronos_pred_val is not None else None
        },
        "sentiment": SENTIMENT_CACHE.get(symbol, {}),
        "event_time_iso": pd.to_datetime(kline_message_data['E'], unit='ms').isoformat()
    }
    # Update the local cache with the new data
    new_kline_for_cache = {"time": live_update_data["kline"]["time"], "open": live_update_data["kline"]["open"], "high": live_update_data["kline"]["high"], "low": live_update_data["kline"]["low"], "close": live_update_data["kline"]["close"], "volume": float(kline_info['v'])}

    if len(HISTORICAL_DATA_CACHE.get(symbol, [])) >= MAX_HISTORICAL_POINTS: HISTORICAL_DATA_CACHE[symbol].pop(0)
    HISTORICAL_DATA_CACHE[symbol].append(new_kline_for_cache)
    LATEST_DATA_CACHE[symbol] = live_update_data 
    socketio.emit('new_kline_data', clean_nan_values(live_update_data))

# callback function that runs after received message
def on_binance_message(ws, message_str):
    try:
        message_data = json.loads(message_str)
        # kline messages
        if 'stream' in message_data and '@kline' in message_data['stream']:
            kline_payload = message_data['data']
            # X is true only if candle is closed.
            if kline_payload['k']['x']:
                # if candle closed, run main prediction 
                process_kline_and_predict(kline_payload)
    except Exception as e:
        print(f"Error processing Binance message: {e}")

# Set up and run websocket connection to binance
def binance_websocket_listener():
    print("Starting Binance WebSocket listener...")
    # Get 1-minute kline stream for all assets
    streams = [f"{coin.lower()}@kline_1m" for coin in ALL_ASSETS]
    SOCKET_URL = "wss://stream.binance.com:9443/stream?streams=" + '/'.join(streams)
    # Create websocket app instance
    ws_app = websocket.WebSocketApp(SOCKET_URL, on_message=on_binance_message)
    ws_app.run_forever()

# Background Scheduler for Historical Data 
# Find last timestamp from csv file 
def get_latest_timestamp(symbol):
    try:
        path = find_asset_data_file(symbol)
        if path:
            df = pd.read_csv(path, usecols=['timestamp']).tail(1)
            if not df.empty:
                return pd.to_datetime(df['timestamp'].iloc[0])
    except Exception as e:
        print(f"[SCHEDULER] Could not read latest timestamp for {symbol}: {e}")
    return None

# Update historical data csv for list of assets
def update_assets_data(asset_list, asset_type):
    print(f"\n[SCHEDULER] Checking for new {asset_type.upper()} data...")

    # Check the correct settings based on assets
    if asset_type == 'stable':
        folder = os.path.join(BASE_DATA_FOLDER, "stable")
        interval = "1hr"
    elif asset_type == 'volatile':
        folder = os.path.join(BASE_DATA_FOLDER, "volatile")
        interval = "5min"
    else:
        return

    for symbol in asset_list:
        # Get last timestamp stored in CSV file
        latest_timestamp = get_latest_timestamp(symbol)
        
        # This is the fully corrected code block
        if latest_timestamp:
            # Use the simple, compatible YYYY-MM-DD HH:MM:SS format
            start_str = (latest_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Use the simple, compatible YYYY-MM-DD format for the initial fetch
            start_str = "2020-01-01"
        # Set end_str to None so API fetches all data up to current
        end_str = None
        
        try:
            # Fetch and save
            data_collect.fetch_and_save_data(
                symbols=[symbol],
                start_date_str=start_str,
                end_date_str=end_str,
                output_folder=folder,
                interval_str=interval
            )
        except Exception as e:
            print(f"[SCHEDULER] CRITICAL ERROR processing {symbol}: {e}")
            
    print(f"[SCHEDULER] Finished checking for {asset_type.upper()} data.")

# Wrapper function to be called stable asset schedule
def trigger_stable_asset_update():
    update_assets_data(STABLE_ASSETS, 'stable')

# Wrapper function to be called by the volatile asset schedule.
def trigger_volatile_asset_update():
    update_assets_data(HIGH_VOLATILITY_ASSETS, 'volatile')

# Function to run scheduling loop
def run_data_collection_scheduler():
    print("[SCHEDULER] Data collection scheduler started with multi-frequency jobs.")
    
    # Schedule for STABLE assets (every hour at XX:01)
    schedule.every().hour.at(":01").do(trigger_stable_asset_update)
    
    # Schedule for VOLATILE assets (every 5 minutes)
    for minute in [f":{m:02d}" for m in range(0, 60, 5)]:
        schedule.every().hour.at(minute).do(trigger_volatile_asset_update)

    # Run jobs once on startup to ensure fresh data
    print("[SCHEDULER] Performing initial data update on startup...")
    trigger_stable_asset_update()
    trigger_volatile_asset_update()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# Routes for User Interface
# Main index page
@app.route('/')
def index_page():
    populate_initial_caches(ALL_ASSETS)
    return render_template('home/index.html', 
                           assets=ALL_ASSETS,
                           asset_names=ASSET_NAMES,
                           initial_data=clean_nan_values(LATEST_DATA_CACHE))

# Dashboard Page
@app.route('/dashboard')
def dashboard_page():
    populate_initial_caches(ALL_ASSETS)
    return render_template('home/dashboard.html', 
                           assets=ALL_ASSETS, 
                           asset_names=ASSET_NAMES,
                           initial_data=clean_nan_values(LATEST_DATA_CACHE))

# Stable Page
@app.route('/stable')
def stable_page():
    populate_initial_caches(STABLE_ASSETS)
    return render_template('home/stable.html', assets=STABLE_ASSETS, 
                                               asset_names=ASSET_NAMES,
                                               initial_data=clean_nan_values(LATEST_DATA_CACHE), 
                                               historical_data=clean_nan_values(HISTORICAL_DATA_CACHE),
                                               format_timestamp=format_iso_timestamp)

# Volatile Page
@app.route('/volatile')
def volatile_page():
    populate_initial_caches(HIGH_VOLATILITY_ASSETS)
    return render_template('home/volatile.html', assets=HIGH_VOLATILITY_ASSETS, 
                                                 asset_names=ASSET_NAMES,
                                                 initial_data=clean_nan_values(LATEST_DATA_CACHE), 
                                                 historical_data=clean_nan_values(HISTORICAL_DATA_CACHE),
                                                 format_timestamp=format_iso_timestamp)


# --- Main Application ---
if __name__ == '__main__':
    # Load al models
    print("="*60 + "\n EXECUTING LATEST VERSION OF THE DASHBOARD   \n" + "="*60)
    load_all_models(ALL_ASSETS)
    load_chronos_models()
    
    #  Preload cache on startup
    print("\n--- Populating historical data caches on startup... ---")
    populate_initial_caches(ALL_ASSETS)

    
    # Get initial sentiment
    print("\n--- Fetching initial sentiment for all assets... ---")
    for asset in ALL_ASSETS:
        socketio.start_background_task(update_single_sentiment, symbol=asset)
        time.sleep(1) 

    # Websocket listener on separate thread
    threading.Thread(target=binance_websocket_listener, daemon=True).start()
    
    # Historical data collection in separate thread
    threading.Thread(target=run_data_collection_scheduler, daemon=True).start()

    # Main Web app
    socketio.run(app, host='0.0.0.0', port=3000, debug=False, allow_unsafe_werkzeug=True)
