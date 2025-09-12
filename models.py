"""
The models.py file is used to train the models and load it in under Model_training
- LightGBM (Quantile Regression) 
- K- Nearest Neighbours (Supertrend) 
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import os
# Import required modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import warnings


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
def train_knn_supertrend_model(file_path, max_retries=3): 
    warnings.filterwarnings('ignore')
   
    # Error Handler to check data file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return None
    try:
        # Get the asset symbol from the file name 
        symbol = os.path.basename(file_path).split('_')[0]
        print(f"\n Training K-NN Model for {symbol}")
        
        # load dataset
        data = pd.read_csv(file_path)
        print(f"{symbol}: Dataset shape: {data.shape}")

        # Data validation and cleaning
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            print(f"{symbol}: Missing required columns. Found: {data.columns}. Skipping.")
            return None

        # Remove rows rows with missing data 
        initial_rows = len(data)
        data.dropna(subset=required_cols, inplace=True)
        print(f"{symbol}: Removed {initial_rows - len(data)} rows with missing data")
        
        # Training need at least 100 rows
        if len(data) < 100: 
            print(f"{symbol}: Not enough data after cleaning ({len(data)} rows). Need at least 100.")
            return None

        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Technical indicators
        data.ta.supertrend(length=10, multiplier=3.0, append=True)
        supertrend_col = next((col for col in data.columns if col.startswith('SUPERT_')), None)
        
        if not supertrend_col:
            print(f"{symbol}: Supertrend calculation failed. Skipping.")
            return None
            
        # Create target with proper time shift (avoid look-ahead bias)
        data['target_close'] = data['close'].shift(-1)
        
        # Feature engineering
        data['price_change'] = data['close'].pct_change()
        data['volatility'] = data['close'].rolling(window=10).std()
        data['volume_ma'] = data['volume'].rolling(window=5).mean()
        
        # Feature columns and remove current close price
        feature_cols = ['open', 'high', 'low', 'volume', 'price_change', 'volatility', 'volume_ma', supertrend_col]
        
        # Clean final dataset
        required_model_cols = feature_cols + ['target_close']
        data.dropna(subset=required_model_cols, inplace=True)
        
        if data.empty:
            print(f"{symbol}: No data left after feature engineering. Skipping.")
            return None

        X = data[feature_cols]
        y = data['target_close']
        print(f"{symbol}: Final dataset shape: X={X.shape}, y={y.shape}")
        
        # Training data
        split_idx = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"{symbol}: Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Automatic parameter tuning
        print(f"{symbol}: Running hyperparameter tuning...")
        
        # adjust KNN search space 
        n_samples = len(X_train)
        k_values = []
        
        if n_samples > 10000:
            k_values = [30, 50, 75, 100, 150, 200]
        elif n_samples > 5000:
            k_values = [20, 30, 50, 75, 100]
        else:
            k_values = [5, 10, 15, 20, 30]
        
        # hyperparameter grids
        param_grid = {
            'knn__n_neighbors': k_values,
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan'],
            'feature_selection__k': ['all', 5, 6, 7]  # Feature selection
        }
        
        # Training pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('knn', KNeighborsRegressor(n_jobs=-1))
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"{symbol}: Best parameters: {best_params}")
        print(f"{symbol}: Best CV score: {grid_search.best_score_:.4f}")
        print(f"{symbol}: Using k={best_params['knn__n_neighbors']} neighbors")
        
        # Model evaluation
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\n--- {symbol} Enhanced K-NN Model Evaluation ---")
        print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test  MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # To detect overfitting by comparing between train and test r squared
        r2_diff = train_r2 - test_r2
        print(f"R² Difference (Train - Test): {r2_diff:.4f}")
        
        # Overfitting thresholds
        severe_overfitting = r2_diff > 0.3 or test_r2 < 0
        moderate_overfitting = r2_diff > 0.15
        
        if severe_overfitting:
            print(f"🚨 Severe overfitting detected for {symbol}!")
            print(f"   - Training R² is too high: {train_r2:.4f}")
            print(f"   - Test R² is poor: {test_r2:.4f}")
            if test_r2 < -0.1:  # Very poor performance
                print(f"❌ Model performance too poor. Not saving model for {symbol}")
                return None
            
        elif moderate_overfitting:
            print(f"⚠️  Moderate overfitting detected for {symbol}")
        else:
            print(f"✅ Good generalization for {symbol}")
        
        # Save the model
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Package model with metadata and components
        scaler = best_model.named_steps['scaler']
        feature_selector = best_model.named_steps['feature_selection']
        knn_model = best_model.named_steps['knn']
        
        model_package = {
            'pipeline': best_model, 
            'scaler': scaler,
            'feature_selector': feature_selector,
            'knn_model': knn_model,
            'feature_columns': feature_cols,
            'selected_features': feature_selector.get_support() if hasattr(feature_selector, 'get_support') else None,
            'symbol': symbol,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_difference': r2_diff,
            'training_timestamp': pd.Timestamp.now(),
            'hyperparameters': best_params,
            'data_shape': X.shape,
            'overfitting_flag': severe_overfitting or moderate_overfitting
        }
        # Add model to disk
        model_filename = os.path.join(OUTPUT_FOLDER, f"{symbol}_knn_supertrend_model.pkl")
        joblib.dump(model_package, model_filename)
        print(f"Enhanced model package for {symbol} saved to [{model_filename}]")
        return model_package
    # error handler
    except Exception as e:
        print(f"Error processing {file_path} for K-NN model: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return None

# To load and to predict the knn model 
def load_and_predict_enhanced_knn(model_path, new_data):
    try:
        # load serialised model package
        model_package = joblib.load(model_path)
        
        # Check if model was overfitted
        if model_package.get('overfitting_flag', False):
            print(f"⚠️  Warning: This model showed signs of overfitting during training")
            print(f"    Test R²: {model_package['test_r2']:.4f}")
        # Get full preprocessing
        pipeline = model_package['pipeline']
        expected_features = model_package['feature_columns']
        
        # ensure all required features are present
        if not all(col in new_data.columns for col in expected_features):
            missing = set(expected_features) - set(new_data.columns)
            raise ValueError(f"Missing features: {missing}")
        
        # Select required features
        X_new = new_data[expected_features]
        predictions = pipeline.predict(X_new)
        
        return {
            'predictions': predictions,
            'model_metadata': {
                'symbol': model_package['symbol'],
                'test_r2': model_package['test_r2'],
                'training_date': model_package['training_timestamp'],
                'overfitting_detected': model_package.get('overfitting_flag', False)
            }
        }
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

# To train the model
def discover_and_train_models(base_folder=BASE_DATA_FOLDER):
    stable_path = os.path.join(base_folder, 'stable')
    volatile_path = os.path.join(base_folder, 'volatile')

    print("\n" + "="*70)
    print("    ENHANCED TRAINING WITH OVERFITTING PREVENTION")
    print("="*70)
    # List of models succeeded and failed
    failed_models = []
    successful_models = []
    
    # Train Stable Assets (K-NN)
    print("\n Process Stable Assets (K-NN)")
    if os.path.exists(stable_path):
        # Loop through csv files in stable dataset folder
        for filename in os.listdir(stable_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(stable_path, filename)
                # get asset symbol
                symbol = filename.split('_')[0]
                print(f"\nProcessing {symbol}...")
                # Train KNN + Supertrend
                result = train_knn_supertrend_model(file_path)
                # success and failure logging
                if result is None:
                    failed_models.append(symbol)
                    print(f"❌ Training failed for {symbol}")
                else:
                    successful_models.append(symbol)
                    print(f"✅ Training completed for {symbol}")
    else:
        print(f"Directory not found: {stable_path}")

    #  Train Volatile Assets (LGBM)
    print("\n Processing Volatile Assets (LGBM)")
    if os.path.exists(volatile_path):
        # loop through csv files in volatile dataset
        for filename in os.listdir(volatile_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(volatile_path, filename)
                # Get asset symbol
                symbol = filename.split('_')[0]
                # Train LightGBM quantile regression model for volatile assets
                train_lgbm_quantile_model(file_path)
                successful_models.append(symbol) # Assume success if it runs without error
    else:
        print(f"Directory not found: {volatile_path}")
    
    # Summary
    print(f"\n" + "="*70)
    print(f"TRAINING SUMMARY")
    print(f"="*70)
    print(f"✅ Successfully trained: {len(successful_models)} models")
    print(f"❌ Failed: {len(failed_models)} models")
    
    if successful_models:
        print(f"Successful: {', '.join(successful_models)}")
    if failed_models:
        print(f"Failed: {', '.join(failed_models)}")

# Usage example
if __name__ == '__main__':
    # Run with automatic hyperparameter tuning
    discover_and_train_models()
