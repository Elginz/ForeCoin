import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import os

# Based and output folders
BASE_DATA_FOLDER = "historic_data"
OUTPUT_FOLDER = "trained_models"

#  LightGBM for Volatile Assets Quantile Regression
def train_lgbm_quantile_model(file_path):
    # Error Handler to ensure data file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return
    try:
        # Get the asset symbol from the file name 
        symbol = os.path.basename(file_path).split('_')[0]
        print(f"\n--- [Volatile Strategy] Training LightGBM Quantile Models for {symbol} ---")
        
        # Load data and set timestamp as index for time-series analysis
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        # Ensure all columns are numeric and remove rows with missing values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)

        # Create another variable for next predicted closing price
        data['target_close'] = data['close'].shift(-1)
        data.dropna(inplace=True)

        # Error handler to ensure data enough data to train after cleaning 
        if data.empty:
            print(f"No data for {symbol}. Skipping.")
            return
        
        X = data[['open', 'high', 'low', 'close', 'volume']]
        y = data['target_close']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Define the quantiles to create a forecast range
        quantiles = {'low': 0.1, 'median': 0.5, 'high': 0.9}
        trained_models = {}

        for name, alpha in quantiles.items():
            print(f"  Training for quantile: {alpha} ({name})...")
            # Create a LightGBM model for each quantile
            model = lgb.LGBMRegressor(
                objective='quantile', 
                # Which quantile to target
                alpha=alpha,
                # Max number of boosting rounds   
                n_estimators=1000,      
                learning_rate=0.05,
                num_leaves=31,
                # To use all CPU cores
                n_jobs=-1,
                random_state=42,
                # Regularisation to prevent overfitting
                colsample_bytree=0.8,   
                subsample=0.8
            )
            # Train the model
            model.fit(X_train, y_train,
                    # Validation set to monitor performance
                      eval_set=[(X_test, y_test)],
                      eval_metric='quantile',
                    # Stop if validation score does not improve for 100 rounds
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            
            trained_models[name] = model
            
            # Evaluate the median model
            if name == 'median':
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                print(f"  Median Model (q=0.5) Test R²: {r2:.4f}")

        # Check if output folder exists
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filename = os.path.join(OUTPUT_FOLDER, f"{symbol}_lgbm_quantile_model.pkl")
        # Save dictionary containing all trained models into a single file
        joblib.dump(trained_models, filename)
        print(f"Quantile models for {symbol} saved as dictionary to [{filename}]")

    except Exception as e:
        print(f"Error processing {file_path} for LGBM Quantile model: {e}")

#  K-Nearest Neighbors for Stable Assets 
def train_knn_supertrend_model(file_path):
    # Error Handler to ensure data file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return
    try:
        # Get the asset symbol from the file name 
        symbol = os.path.basename(file_path).split('_')[0]
        print(f"\n--- [Stable Strategy] Training K-NN Supertrend Model for {symbol} ---")
        data = pd.read_csv(file_path)

        # Check the data types of all columns 
        print(f"{symbol}: Checking initial data types and info...")
        data.info()

        # Check that needed columns are present in the dataset
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"{symbol}: Missing required columns. Found: {data.columns}. Skipping.")
            return

        # Remove any rows with missing data 
        data.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Check that there is enough data for training
        if len(data) < 30:
            print(f"{symbol}: Not enough data after initial cleaning ({len(data)} rows). Skipping.")
            return

        # Convert the timestamp to a datetime object
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Calculate the Supertrend indicator 
        data.ta.supertrend(length=10, multiplier=3.0, append=True)
        
        # Find name of Supertrend column 
        supertrend_col = next((col for col in data.columns if col.startswith('SUPERT_')), None)
        # Create another variable for next predicted closing price
        data['target_close'] = data['close'].shift(-1)
        
        # Check if calculation was successful
        if not supertrend_col:
            print(f"{symbol}: Supertrend column not found after calculation. Skipping.")
            return
        
        # Features for training
        feature_cols = ['open', 'high', 'low', 'close', 'volume', supertrend_col]
        # Remove missing values
        required_model_cols = feature_cols + ['target_close']
        data.dropna(subset=required_model_cols, inplace=True)

        # Check if data is empty
        if data.empty:
            print(f"{symbol}: No data left after final cleaning. Skipping.")
            return

        X = data[feature_cols]
        y = data['target_close']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # KNN regressor model
        model = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # To calculate MSE and R-squared
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"--- {symbol} K-NN Model Evaluation ---")
        print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test  MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

        # Save trained models
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

    #  Training Stable Asset Models
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
                train_lgbm_quantile_model(file_path)
    else:
        print(f"Directory not found: {volatile_path}")

# Main Process
if __name__ == '__main__':
    discover_and_train_models()
