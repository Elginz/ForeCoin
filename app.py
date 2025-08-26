### app.py ###

from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import os
import joblib 
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import time
import torch 
import math 
import websocket
import json
import glob

# --- Import local python files ---
from predict_chronos import forecast_next_hour
from webscraper import determine_sentiment, BTCUSDT as BTC_QUERY, ETHUSDT as ETH_QUERY, BNBUSDT as BNB_QUERY, DOGEUSDT as DOGE_QUERY, SHIBUSDT as SHIB_QUERY

# --- App Configuration and Global Variables ---
template_folder = os.path.join(os.getcwd(), 'apps', 'templates')
static_folder = os.path.join(os.getcwd(), 'apps', 'static')
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.config['SECRET_KEY'] = 'your_secret_key_123!' 
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Asset lists and constants ---
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
ALL_ASSETS = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS
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

# --- Asset Query Mappings ---
ASSET_QUERIES = { 'BTCUSDT': BTC_QUERY, 'ETHUSDT': ETH_QUERY, 
                  'BNBUSDT': BNB_QUERY, 'DOGEUSDT': DOGE_QUERY, 
                  'SHIBUSDT': SHIB_QUERY }

# --- Helper Functions ---
def find_asset_data_file(symbol):
    subfolder = 'stable' if symbol in STABLE_ASSETS else 'volatile'
    simple_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_data.csv")
    if os.path.exists(simple_path):
        return simple_path
    search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def format_iso_timestamp(iso_string):
    if not iso_string or 'T' not in str(iso_string): return 'N/A'
    try:
        dt_object = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt_object.strftime('%Y-%m-%d (%H:%M:%S UTC)')
    except (ValueError, TypeError): return 'N/A'

def clean_nan_values(obj):
    if isinstance(obj, dict): return {k: clean_nan_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_nan_values(i) for i in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None
    return obj

def populate_initial_caches(symbols):
    for symbol in symbols:
        if not HISTORICAL_DATA_CACHE[symbol]:
            try:
                path = find_asset_data_file(symbol)
                if path and os.path.exists(path):
                    df = pd.read_csv(path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df_recent = df.tail(MAX_HISTORICAL_POINTS)
                    
                    for _, row in df_recent.iterrows():
                        HISTORICAL_DATA_CACHE[symbol].append({ "time": row['timestamp'].timestamp(), "open": row['open'], "high": row['high'], "low": row['low'], "close": row['close'] })
                    
                    if not LATEST_DATA_CACHE.get(symbol) and not df_recent.empty:
                        last_row = df_recent.iloc[-1]
                        last_kline = { "time": last_row['timestamp'].timestamp(), "open": last_row['open'], "high": last_row['high'], "low": last_row['low'], "close": last_row['close'] }
                        LATEST_DATA_CACHE[symbol] = { "symbol": symbol, "kline": last_kline, "prediction": {}, "sentiment": {}, "event_time_iso": last_row['timestamp'].isoformat() }
            except Exception as e:
                print(f"Error populating caches for {symbol}: {e}")

# --- Model Loading ---
def load_all_models(symbols):
    global MODELS_DICT
    models_dir = "trained_models"
    for symbol in symbols:
        if symbol in STABLE_ASSETS:
            model_type = "KNN Supertrend"
            model_filename = f"{symbol}_knn_supertrend_model.pkl"
        else:
            model_type = "LGBM Quantile"
            model_filename = f"{symbol}_lgbm_quantile_model.pkl"

        model_path = os.path.join(models_dir, model_filename)
        print(f"Attempting to load {model_type} model for {symbol}...")
        try:
            if os.path.exists(model_path):
                MODELS_DICT[symbol] = joblib.load(model_path)
                print(f"--> SUCCESS: {symbol}'s {model_type} model loaded.")
            else:
                print(f"--> WARNING: Model file not found at '{model_path}'")
                MODELS_DICT[symbol] = None
        except Exception as e:
            print(f"--> ERROR: Could not load model for {symbol}: {e}")
            MODELS_DICT[symbol] = None

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
                df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
                values = df["close"].values
                context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
                pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
                CHRONOS_MODELS[symbol] = {"pipeline": pipeline, "context": context, "df": df}
                print(f"Successfully loaded Chronos model for {symbol}.")
            except Exception as e:
                print(f"Error loading Chronos model for {symbol}: {e}"); CHRONOS_MODELS[symbol] = None
    except ImportError:
        print("Chronos library not found. Skipping Chronos model loading.")

# --- Background Threads ---
def update_single_sentiment(symbol):
    with SENTIMENT_LOCK:
        query = ASSET_QUERIES.get(symbol)
        if not query: return
        
        print(f"--- Updating sentiment for {symbol}... ---")
        try:
            # --- FIX: Unpack the two return values from the new webscraper ---
            final_score, final_confidence = determine_sentiment(query)
            
            # --- FIX: Determine the label based on the score ---
            label = "neutral"
            if final_score > 0.05:
                label = "positive"
            elif final_score < -0.05:
                label = "negative"
            
            sentiment_data = {"label": label, "score": final_score, "confidence": final_confidence}
            SENTIMENT_CACHE[symbol] = sentiment_data
            LAST_SENTIMENT_UPDATE[symbol] = datetime.now()
            
            socketio.emit('sentiment_update', {'symbol': symbol, 'sentiment': sentiment_data})
            print(f"Sentiment for {symbol}: {label} (Score: {final_score:.2f}) - Pushed to clients.")
        except Exception as e:
            print(f"An error occurred in sentiment update for {symbol}: {e}")

# --- Prediction Logic ---
def process_kline_and_predict(kline_message_data):
    kline_info = kline_message_data['k']
    symbol = kline_info['s']
    
    last_update = LAST_SENTIMENT_UPDATE.get(symbol)
    if not last_update or (datetime.now() - last_update) > timedelta(minutes=30):
        if not SENTIMENT_LOCK.locked():
             socketio.start_background_task(update_single_sentiment, symbol=symbol)

    knn_pred_val = None
    lgbm_preds = {'low': None, 'median': None, 'high': None}
    chronos_pred_val = None
    
    model_or_models = MODELS_DICT.get(symbol)
    if model_or_models:
        try:
            features_dict = {
                'open': float(kline_info['o']), 'high': float(kline_info['h']),
                'low': float(kline_info['l']), 'close': float(kline_info['c']),
                'volume': float(kline_info['v'])
            }
            
            if symbol in STABLE_ASSETS:
                history_df = pd.DataFrame(HISTORICAL_DATA_CACHE.get(symbol, []))
                if history_df.shape[0] > 20: 
                    current_kline_df = pd.DataFrame([features_dict])
                    combined_df = pd.concat([history_df, current_kline_df], ignore_index=True)

                    indicator_df = ta.supertrend(high=combined_df['high'], low=combined_df['low'], close=combined_df['close'], length=10, multiplier=3.0)
                    
                    if indicator_df is not None and not indicator_df.empty:
                        supertrend_col_name = next((col for col in indicator_df.columns if col.startswith('SUPERT_')), None)
                        
                        if supertrend_col_name:
                            last_supertrend_val = indicator_df[supertrend_col_name].iloc[-1]
                            
                            if not pd.isna(last_supertrend_val):
                                features_dict[supertrend_col_name] = last_supertrend_val
                                feature_order = ['open', 'high', 'low', 'close', 'volume', supertrend_col_name]
                                features_df = pd.DataFrame([features_dict], columns=feature_order)
                                knn_pred_val = model_or_models.predict(features_df)[0]
            
            elif symbol in HIGH_VOLATILITY_ASSETS and isinstance(model_or_models, dict):
                feature_order = ['open', 'high', 'low', 'close', 'volume']
                features_df = pd.DataFrame([features_dict], columns=feature_order)
                
                for name, model in model_or_models.items():
                    lgbm_preds[name] = model.predict(features_df)[0]

        except Exception as e:
            print(f"--> UNEXPECTED ERROR during model prediction for {symbol}: {e}")

    if symbol in CHRONOS_MODELS and CHRONOS_MODELS[symbol] is not None:
        try:
            model_info = CHRONOS_MODELS[symbol]
            new_price_tensor = torch.tensor([float(kline_info['c'])], dtype=model_info["context"].dtype)
            model_info["context"] = torch.cat([model_info["context"], new_price_tensor.unsqueeze(0)], dim=1)[:, -1024:]
            _, median, _, _ = forecast_next_hour(model_info["df"], model_info["df"]["close"].values, model_info["context"], model_info["pipeline"])
            chronos_pred_val = median[0]
        except Exception as e:
            print(f"ERROR making Chronos prediction for {symbol}: {e}")

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
    
    if len(HISTORICAL_DATA_CACHE.get(symbol, [])) >= MAX_HISTORICAL_POINTS: HISTORICAL_DATA_CACHE[symbol].pop(0)
    HISTORICAL_DATA_CACHE[symbol].append(live_update_data["kline"])
    LATEST_DATA_CACHE[symbol] = live_update_data 
    socketio.emit('new_kline_data', clean_nan_values(live_update_data))

def on_binance_message(ws, message_str):
    try:
        message_data = json.loads(message_str)
        if 'stream' in message_data and '@kline' in message_data['stream']:
            kline_payload = message_data['data']
            if kline_payload['k']['x']:
                process_kline_and_predict(kline_payload)
    except Exception as e:
        print(f"Error processing Binance message: {e}")

def binance_websocket_listener():
    print("Starting Binance WebSocket listener...")
    streams = [f"{coin.lower()}@kline_1m" for coin in ALL_ASSETS]
    SOCKET_URL = "wss://stream.binance.com:9443/stream?streams=" + '/'.join(streams)
    ws_app = websocket.WebSocketApp(SOCKET_URL, on_message=on_binance_message)
    ws_app.run_forever()

@app.route('/')
def dashboard_page():
    populate_initial_caches(ALL_ASSETS)
    return render_template('home/dashboard.html', assets=ALL_ASSETS, initial_data=clean_nan_values(LATEST_DATA_CACHE))

@app.route('/stable')
def stable_page():
    populate_initial_caches(STABLE_ASSETS)
    return render_template('home/stable.html', assets=STABLE_ASSETS, 
                                               initial_data=clean_nan_values(LATEST_DATA_CACHE), 
                                               historical_data=clean_nan_values(HISTORICAL_DATA_CACHE),
                                               format_timestamp=format_iso_timestamp)

@app.route('/volatile')
def volatile_page():
    populate_initial_caches(HIGH_VOLATILITY_ASSETS)
    return render_template('home/volatile.html', assets=HIGH_VOLATILITY_ASSETS, 
                                                 initial_data=clean_nan_values(LATEST_DATA_CACHE), 
                                                 historical_data=clean_nan_values(HISTORICAL_DATA_CACHE),
                                                 format_timestamp=format_iso_timestamp)

if __name__ == '__main__':
    print("="*60 + "\n   EXECUTING LATEST VERSION OF THE DASHBOARD   \n" + "="*60)
    load_all_models(ALL_ASSETS)
    load_chronos_models()
    
    print("\n--- Fetching initial sentiment for all assets... ---")
    for asset in ALL_ASSETS:
        socketio.start_background_task(update_single_sentiment, symbol=asset)
        time.sleep(1) 

    threading.Thread(target=binance_websocket_listener, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=3000, debug=False, allow_unsafe_werkzeug=True)

