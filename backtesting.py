"""
This is a backtesting.py file 

It runs on a schedule (every hour at 30 minutes past the hour).
It checks the performance of the current production models. 
If a model's recent prediction accuracy drops below a defined threshold, it automatically
triggers a retraining of that model using the functions

imported from models.py.
"""

import pandas as pd
import numpy as np
import joblib
import os
import pandas_ta as ta
import glob
from datetime import datetime
import schedule
import time
import json
import warnings

# training functions from models.py 
from models import train_knn_supertrend_model, train_lgbm_quantile_model

warnings.filterwarnings('ignore')

# Paths for data, trained models and performance logs
BASE_DATA_FOLDER = "historic_data"
MODELS_FOLDER = "trained_models"
LOG_FOLDER = "prediction_logs"
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
ALL_ASSETS = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS

# --- Backtesting Parameters ---
#TO check accuracy over the last 5 predictions.
VALIDATION_WINDOW = 5
# Must have at least 3/5 (60%) correct predictions.
ACCURACY_THRESHOLD = 0.6  

# This class is used to predict generation, validation, logging and retraining triggers
class ModelValidator:
    # initialises modelValidator
    def __init__(self):
        self.prediction_logs = {symbol: [] for symbol in ALL_ASSETS}
        os.makedirs(LOG_FOLDER, exist_ok=True)
        self.log_file_path = os.path.join(LOG_FOLDER, "backtest_log.json")
        self.load_logs()

    # To load previous validation logs form a file
    def load_logs(self):
        # If the path exists
        if os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'r') as f:
                    self.prediction_logs = json.load(f)
                print("Successfully loaded existing prediction logs.")
            # Error handler
            except Exception as e:
                print(f"Could not load logs, starting fresh. Error: {e}")
        else:
            print("No existing log file found. Starting fresh.")
    # To save the logs
    def save_logs(self):
        """Saves the current validation logs to a file."""
        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(self.prediction_logs, f, indent=4)
        except Exception as e:
            print(f"Error saving prediction logs: {e}")
    # Find the latest data csv 
    def find_asset_data_file(self, symbol):
        subfolder = 'stable' if symbol in STABLE_ASSETS else 'volatile'
        search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
        files = glob.glob(search_path)
        return max(files, key=os.path.getmtime) if files else None
    
    def make_prediction(self, symbol):
        """
        Generates a new prediction for an asset using its current production model.
        This logic is adapted from your app.py to be compatible.
        """
        try:
            model_type = "knn_supertrend" if symbol in STABLE_ASSETS else "lgbm_quantile"
            model_path = os.path.join(MODELS_FOLDER, f"{symbol}_{model_type}_model.pkl")
            if not os.path.exists(model_path):
                print(f"Model not found for {symbol}. Skipping prediction.")
                return None, None

            model_package = joblib.load(model_path)
            
            file_path = self.find_asset_data_file(symbol)
            if not file_path: return None, None
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            
            # Use last 200 rows for feature calculation
            latest_data = df.tail(200).copy()
            if len(latest_data) < 20: return None, None
            
            current_price = float(latest_data.iloc[-1]['close'])

            if symbol in STABLE_ASSETS:
                # Replicate K-NN feature engineering from models.py
                latest_data.ta.supertrend(length=10, multiplier=3.0, append=True)
                latest_data['price_change'] = latest_data['close'].pct_change()
                latest_data['volatility'] = latest_data['close'].rolling(window=10).std()
                latest_data['volume_ma'] = latest_data['volume'].rolling(window=5).mean()
                latest_data.ffill(inplace=True)
                latest_data.dropna(inplace=True)

                if latest_data.empty:
                    print(f"Not enough data for {symbol} after feature engineering. Skipping.")
                    return None, None
                
                # ===== START FIX =====
                # Get the exact list of feature columns the model was trained on
                expected_features = model_package.get('feature_columns')
                if not expected_features:
                    print(f"ERROR: 'feature_columns' not found in model package for {symbol}.")
                    return None, None
                
                # Select ONLY those features from the latest data point
                features_for_prediction = latest_data.tail(1)[expected_features]
                # ===== END FIX =====
                
                pipeline = model_package.get('pipeline')
                predicted_price = pipeline.predict(features_for_prediction)[0]
                return current_price, float(predicted_price)
            
            elif symbol in HIGH_VOLATILITY_ASSETS:
                # LGBM prediction
                features = latest_data.tail(1)[['open', 'high', 'low', 'close', 'volume']]
                lgbm_median_model = model_package['median']
                predicted_price = lgbm_median_model.predict(features)[0]
                return current_price, float(predicted_price)
        
        except Exception as e:
            print(f"Error making prediction for {symbol}: {e}")
        return None, None

    # Runs the cycle at every 30 minutes of an hour
    def run_validation_cycle(self):
        print(f"\n--- Running Validation Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")   
        for symbol in ALL_ASSETS:
            print(f"\nProcessing {symbol}...")
            
            # Generate new prediction and get the price
            current_price, predicted_price = self.make_prediction(symbol)
            
            if current_price is None or predicted_price is None:
                continue
            
            # Check the previous prediction from the last cycle
            # Check if a log list exists and is not empty
            if self.prediction_logs.get(symbol):
                last_log = self.prediction_logs[symbol][-1]
                if last_log['actual_price_later'] is None:
                    last_log['actual_price_later'] = current_price
                    actual_direction = 'up' if current_price > last_log['price_at_prediction'] else 'down'
                    last_log['actual_direction'] = actual_direction
                    last_log['is_correct'] = (last_log['predicted_direction'] == actual_direction)
                    print(f"  Validated previous prediction: Predicted {last_log['predicted_direction']}, Actual was {actual_direction}. Correct: {last_log['is_correct']}")

            # Log the new prediction
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "price_at_prediction": current_price,
                "predicted_price": predicted_price,
                "predicted_direction": 'up' if predicted_price > current_price else 'down',
                "actual_price_later": None,
                "actual_direction": None,
                "is_correct": None
            }
            self.prediction_logs.setdefault(symbol, []).append(log_entry)
            
            # Check if need to retrain
            validated_logs = [log for log in self.prediction_logs.get(symbol, []) if log['is_correct'] is not None]
            # Only check accuracy if have enough predictions
            if len(validated_logs) >= VALIDATION_WINDOW:
                recent_logs = validated_logs[-VALIDATION_WINDOW:]
                correct_count = sum(1 for log in recent_logs if log['is_correct'])
                accuracy = correct_count / len(recent_logs)
                
                print(f"  Recent accuracy over last {len(recent_logs)} predictions: {accuracy:.0%} ({correct_count}/{len(recent_logs)})")
                # when accuracy below threshold
                if accuracy < ACCURACY_THRESHOLD:
                    print(f"  Accuracy is below {ACCURACY_THRESHOLD:.0%}. Triggering retraining...")
                    self.trigger_retrain(symbol)
                else:
                    print(f"  Model performance is satisfactory. No retraining needed.")
            else:
                print(f"  Not enough validated predictions to check accuracy ({len(validated_logs)}/{VALIDATION_WINDOW}).")

        # Save logs to disk after processing all assets
        self.save_logs()

    # Calls the appropriate training function from models.py
    def trigger_retrain(self, symbol):
        file_path = self.find_asset_data_file(symbol)
        if not file_path:
            print(f"  Could not find data file for {symbol} to retrain.")
            return

        print(f"Starting robust retraining for {symbol}...")
        try:
            if symbol in STABLE_ASSETS:
                train_knn_supertrend_model(file_path)
            else:
                train_lgbm_quantile_model(file_path)
            
            # After retraining, clear the logs to refresh
            self.prediction_logs[symbol] = []
            print(f"  --- Retraining complete for {symbol}. Prediction log has been reset. ---")
        except Exception as e:
            print(f"  An error occurred during retraining: {e}")

def main():
    """Main function to set up and run the schedule."""
    validator = ModelValidator()
    
    print("="*80)
    print(" Automated Model Backtester and Retraining Service")
    print(f" -> This script will run every hour at XX:30")
    print(f" -> It will check the last {VALIDATION_WINDOW} predictions for each model.")
    print(f" -> If accuracy is below {ACCURACY_THRESHOLD:.0%}, it will trigger a full retrain.")
    print("="*80)

    # Schedule the job
    schedule.every().hour.at(":30").do(validator.run_validation_cycle)
    
    # Run once on startup to initialize
    print("Running initial validation cycle on startup...")
    validator.run_validation_cycle()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
