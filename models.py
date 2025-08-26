import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor # No longer needed for volatile assets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb # Import LightGBM
import joblib
import os

# files 
BASE_DATA_FOLDER = "historic_data"
OUTPUT_FOLDER = "trained_models"


#  LightGBM for Volatile Assets Quantile Regression
def train_lgbm_quantile_model(file_path):
    """
    Trains a set of LightGBM models for quantile regression (10th, 50th, 90th percentiles).
    Saves a dictionary of trained models to a single .pkl file.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return
    try:
        symbol = os.path.basename(file_path).split('_')[0]
        print(f"\n--- [Volatile Strategy] Training LightGBM Quantile Models for {symbol} ---")
        
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)
        data['target_close'] = data['close'].shift(-1)
        data.dropna(inplace=True)

        if data.empty:
            print(f"No data for {symbol}. Skipping.")
            return

        X = data[['open', 'high', 'low', 'close', 'volume']]
        y = data['target_close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Define the quantiles we want to predict
        quantiles = {'low': 0.1, 'median': 0.5, 'high': 0.9}
        trained_models = {}

        for name, alpha in quantiles.items():
            print(f"  Training for quantile: {alpha} ({name})...")
            # Create and train a LightGBM model for each quantile
            model = lgb.LGBMRegressor(
                objective='quantile',  # Specify quantile regression
                alpha=alpha,            # The quantile to predict
                n_estimators=1000,      # Increased estimators for better accuracy
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=-1,
                random_state=42,
                colsample_bytree=0.8,   # Add some regularization
                subsample=0.8
            )
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric='quantile',
                      callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping
            
            trained_models[name] = model
            
            # Evaluate the median model for reference
            if name == 'median':
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                print(f"  Median Model (q=0.5) Test R²: {r2:.4f}")

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        # The filename now reflects the new model type
        filename = os.path.join(OUTPUT_FOLDER, f"{symbol}_lgbm_quantile_model.pkl")
        joblib.dump(trained_models, filename)
        print(f"Quantile models for {symbol} saved as dictionary to [{filename}]")

    except Exception as e:
        print(f"Error processing {file_path} for LGBM Quantile model: {e}")

#  K-Nearest Neighbors for Stable Assets (This function remains unchanged)
def train_knn_supertrend_model(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return
    try:
        symbol = os.path.basename(file_path).split('_')[0]
        print(f"\n--- [Stable Strategy] Training K-NN Supertrend Model for {symbol} ---")
        data = pd.read_csv(file_path)

        # Check the data types of all columns 
        print(f"{symbol}: Checking initial data types and info...")
        data.info()

        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"{symbol}: Missing required columns. Found: {data.columns}. Skipping.")
            return

        data.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        if len(data) < 30:
            print(f"{symbol}: Not enough data after initial cleaning ({len(data)} rows). Skipping.")
            return

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        #  Feature Engineering 
        data.ta.supertrend(length=10, multiplier=3.0, append=True)
        supertrend_col = next((col for col in data.columns if col.startswith('SUPERT_')), None)
        data['target_close'] = data['close'].shift(-1)
        
        if not supertrend_col:
            print(f"{symbol}: Supertrend column not found after calculation. Skipping.")
            return
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', supertrend_col]
        required_model_cols = feature_cols + ['target_close']
        data.dropna(subset=required_model_cols, inplace=True)

        if data.empty:
            print(f"{symbol}: No data left after final cleaning. Skipping.")
            return

        X = data[feature_cols]
        y = data['target_close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"--- {symbol} K-NN Model Evaluation ---")
        print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test  MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        model_filename = os.path.join(OUTPUT_FOLDER, f"{symbol}_knn_supertrend_model.pkl")
        joblib.dump(model, model_filename)
        print(f"Model for {symbol} saved as [{model_filename}]")

    except Exception as e:
        print(f"Error processing {file_path} for K-NN model: {e}")


#  Main Function 
def discover_and_train_models(base_folder=BASE_DATA_FOLDER):
    stable_path = os.path.join(base_folder, 'stable')
    volatile_path = os.path.join(base_folder, 'volatile')

    #  Train Stable Asset Models
    print("\n" + "="*54 + "\n       DISCOVERING & TRAINING STABLE ASSETS        \n" + "="*54)
    if os.path.exists(stable_path):
        for filename in os.listdir(stable_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(stable_path, filename)
                train_knn_supertrend_model(file_path)
    else:
        print(f"Directory not found: {stable_path}")

    # Train Volatile Asset Models 
    print("\n" + "="*54 + "\n   DISCOVERING & TRAINING HIGH-VOLATILITY ASSETS   \n" + "="*54)
    if os.path.exists(volatile_path):
        for filename in os.listdir(volatile_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(volatile_path, filename)
                # **MODIFIED: Call the new LGBM function instead of the old RF function**
                train_lgbm_quantile_model(file_path)
    else:
        print(f"Directory not found: {volatile_path}")


# Main Process
if __name__ == '__main__':
    discover_and_train_models()
