"""
This script connects to the Binance WebSocket to receive live 1-hour candlestick data 
for BTCUSDT, ETHUSDT, or BNBUSDT

It then uses a pre-trained model to predict the closing price of the *next* hour.

It also has a daily scheduled task to run `data_collect.py` to update historical data.
"""

import json
import websocket
import pandas as pd
import joblib
import numpy as np
import os
import subprocess
import schedule
import time
import threading
import pytz
from datetime import datetime


# Assets to track  and construct the WebSocket stream URL
assets_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
# Stream name
kline_streams = [coin.lower() + '@kline_1h' for coin in assets_symbols]
socket_streams_param = '/'.join(kline_streams)
socket_url = "wss://stream.binance.com:9443/stream?streams=" + socket_streams_param

# Load pre-trained models for BTCUSDT, ETHUSDT, BNBUSDT 
def load_models(symbols):
    loaded_models = {}
    # Folder for trained models
    models_dir = "trained_models"
    for symbol in symbols:
        # File path for each model
        model_path = os.path.join(models_dir, f"{symbol}_rfr_model.pkl")
        try:
            # Load the model
            model = joblib.load(model_path)
            loaded_models[symbol] = model
            print(f"Model for {symbol} loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found for {symbol}")
            loaded_models[symbol] = None
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
            loaded_models[symbol] = None
    return loaded_models

# Load models once globally
print("Loading models...")
models_dict = load_models(assets_symbols)

# Function to process kline and predict
# Processes a closed candlestick received from binance websocket and uses a pretrained model to predict 
# the next closing price of a cryptocurrency, then prints a trend signal based on that prediction

def process_kline_and_predict(kline_message_data):
    # Extracts the actual kline info from the websocket message
    kline_info = kline_message_data['k']
    # The symbol could be BTCUSDT or ETHUSDT, etc
    symbol = kline_info['s']
    open_price = float(kline_info['o'])
    high_price = float(kline_info['h'])
    low_price = float(kline_info['l'])
    close_price = float(kline_info['c'])
    volume = float(kline_info['v'])
    # Convert the timestamp from miliseconds to readable datetime
    event_time = pd.to_datetime(kline_message_data['E'], unit='ms')
    kline_close_time = pd.to_datetime(kline_info['T'], unit='ms')
    # Convert the numerical values into a numpy array as a 2d array
    features = np.array([[open_price, high_price, low_price, close_price, volume]])
    # Check if model exist 
    if symbol in models_dict and models_dict[symbol] is not None:
        model = models_dict[symbol]
        try:
            # Use the model to predict the next closing price
            predicted_close = model.predict(features)[0]
            print(f"[{event_time}] {symbol}: Close={close_price:.2f}, Predicted Next Close={predicted_close:.2f}")
            # Generate simple trading signal based on prediction
            if predicted_close > close_price:
                print("  Signal: UP trend")
                print("  Trading Advice: BUY")
            elif predicted_close < close_price:
                print("  Signal: DOWN trend")
                print("  Trading Advice: SELL")
            else:
                print("  Signal: SIDEWAYS")
        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
    else:
        print(f"No model available for {symbol}")


# WebSocket callbacks
def on_open(ws):
    print("[WebSocket Opened]")

# Error handlers
def on_error(ws, error):
    print(f"[WebSocket Error] {error}")

def on_close(ws, code, msg):
    print(f"[WebSocket Closed] Code: {code}, Msg: {msg}")

# Whenever a mesage is received from websocket
def on_message(ws, msg_str):
    try:
        data = json.loads(msg_str)
        # checks if message is a 1 hour update
        if 'stream' in data and '@kline_1h' in data['stream']:
            kline_data = data['data']
            if kline_data['k']['x']:
                print(f"\n--- Kline closed for {kline_data['k']['s']} ---")
                # if Kline closed, make a prediction
                process_kline_and_predict(kline_data)
    except Exception as e:
        print(f"Error processing message: {e}")

# Scheduled function
def update_and_run_data_collect():
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    end_date = now_est.strftime('%Y-%m-%d %H:%M:%S')
    os.environ['END_DATE'] = end_date
    print(f"\n[Scheduler] Running data_collect.py with END_DATE={end_date}")
    try:
        # run data_collect.py as a separate process
        subprocess.run(['python', 'data_collect.py'], check=True)
        print("[Scheduler] data_collect.py executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[Scheduler Error] {e}")

# Fucntion to schedule data collection to 12:00 everyday
def run_scheduler():
    schedule.every().day.at("12:00").do(update_and_run_data_collect)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":

    models_dict = load_models(assets_symbols)

    # Start scheduler in background
    threading.Thread(target=run_scheduler, daemon=True).start()

    print(f"Connecting to WebSocket: {socket_url}")
    ws_app = websocket.WebSocketApp(socket_url,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
    ws_app.run_forever()
