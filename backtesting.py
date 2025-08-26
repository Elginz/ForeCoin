"""
This script is used to evaluate and backtest the models, aligning with the objectives
set in the preliminary project report. It runs separate backtests for the primary
model (KNN/LGBM), Chronos T5, and Sentiment Analysis, and calculates detailed
statistical and financial metrics for each.
"""

import pandas as pd
import numpy as np
import joblib
import os
import pandas_ta as ta
import glob
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from tqdm import tqdm
import time
import torch

# --- Import local python files ---
from predict_chronos import forecast_next_hour
# Assuming webscraper can be used to simulate sentiment
from webscraper import determine_sentiment

# --- Configuration ---
BASE_DATA_FOLDER = "historic_data"
MODELS_FOLDER = "trained_models"
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']

# --- Simulation Parameters ---
INITIAL_CAPITAL = 10000
TRADE_SIZE_PERCENT = 0.5
TRANSACTION_COST_PERCENT = 0.001

# --- Helper Functions ---
def find_asset_data_file(symbol):
    subfolder = 'stable' if symbol in STABLE_ASSETS else 'volatile'
    search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
    files = glob.glob(search_path)
    if not files: return None
    return max(files, key=os.path.getmtime)

def load_data_and_models(symbol):
    print(f"\n--- Loading data and all models for {symbol} ---")
    file_path = find_asset_data_file(symbol)
    if not file_path:
        print(f"  - Data file not found for {symbol}. Skipping.")
        return None, None, None
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    model_type = "knn_supertrend" if symbol in STABLE_ASSETS else "lgbm_quantile"
    model_path = os.path.join(MODELS_FOLDER, f"{symbol}_{model_type}_model.pkl")
    if not os.path.exists(model_path):
        print(f"  - Primary model file not found at '{model_path}'. Skipping.")
        return None, None, None
    primary_model = joblib.load(model_path)

    try:
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny", device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        chronos_model = {"pipeline": pipeline}
        print("  - Data and all models loaded successfully.")
        return df, primary_model, chronos_model
    except ImportError:
        print("  - Chronos library not found. Skipping Chronos model loading.")
        return df, primary_model, None

# --- Signal Generation Functions ---
def generate_primary_signals(df, primary_model, symbol):
    df_copy = df.copy()
    df_copy['signal'] = 0
    if symbol in STABLE_ASSETS:
        df_copy.ta.supertrend(length=10, multiplier=3.0, append=True)
        supertrend_col = next((col for col in df_copy.columns if col.startswith('SUPERT_')), None)
        if supertrend_col:
            feature_cols = ['open', 'high', 'low', 'close', 'volume', supertrend_col]
            X = df_copy[feature_cols].dropna()
            if not X.empty:
                df_copy.loc[X.index, 'prediction'] = primary_model.predict(X)
                buy = (df_copy['prediction'] > df_copy['close']) & (df_copy['close'] > df_copy[supertrend_col])
                sell = (df_copy['prediction'] < df_copy['close']) & (df_copy['close'] < df_copy[supertrend_col])
                df_copy.loc[buy, 'signal'] = 1
                df_copy.loc[sell, 'signal'] = -1
    elif symbol in HIGH_VOLATILITY_ASSETS:
        X = df_copy[['open', 'high', 'low', 'close', 'volume']]
        df_copy['pred_low'] = primary_model['low'].predict(X)
        df_copy['pred_median'] = primary_model['median'].predict(X)
        df_copy['pred_high'] = primary_model['high'].predict(X)
        df_copy.loc[df_copy['close'] < df_copy['pred_low'], 'signal'] = 1
        df_copy.loc[df_copy['close'] > df_copy['pred_high'], 'signal'] = -1
    return df_copy

def precompute_chronos_predictions(df, chronos_model):
    print("Pre-computing Chronos T5 predictions for the backtest period...")
    predictions = []
    pipeline = chronos_model['pipeline']
    context_size = 256
    for i in tqdm(range(context_size, len(df)), desc="  - Forecasting with Chronos"):
        context = torch.tensor(df['close'].values[i-context_size:i], dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, 1)
        median_forecast = np.quantile(forecast[0].numpy(), 0.5, axis=0)[0]
        predictions.append(median_forecast)
    full_predictions = [np.nan] * context_size + predictions
    return pd.Series(full_predictions, index=df.index)

def generate_chronos_signals(df):
    df_copy = df.copy()
    df_copy['signal'] = np.where(df_copy['chronos_pred'] > df_copy['close'], 1, -1)
    df_copy.loc[df_copy['chronos_pred'].isna(), 'signal'] = 0
    return df_copy

def simulate_sentiment_signals(df):
    print("Simulating sentiment signals...")
    df_copy = df.copy()
    sentiments = np.random.choice([-1, 0, 1], size=len(df_copy), p=[0.3, 0.4, 0.3])
    df_copy['signal'] = sentiments
    return df_copy

# --- Backtesting and Metrics Calculation ---
def run_backtest(df_with_signals):
    capital = INITIAL_CAPITAL
    position_size, position = 0, 0
    portfolio_values, trades = [], []
    for i in range(len(df_with_signals)):
        row = df_with_signals.iloc[i]
        if row['signal'] == 1 and position == 0:
            trade_amount = capital * TRADE_SIZE_PERCENT
            position_size = trade_amount / row['close']
            capital -= position_size * row['close'] * (1 + TRANSACTION_COST_PERCENT)
            position = 1
            trades.append({'entry_price': row['close']})
        elif row['signal'] == -1 and position == 1:
            capital += position_size * row['close'] * (1 - TRANSACTION_COST_PERCENT)
            position, position_size = 0, 0
            if trades and 'exit_price' not in trades[-1]:
                last_trade = trades[-1]
                last_trade['exit_price'] = row['close']
                last_trade['profit_pct'] = (last_trade['exit_price'] - last_trade['entry_price']) / last_trade['entry_price']
        current_value = capital + (position_size * row['close'] if position == 1 else 0)
        portfolio_values.append(current_value)
    return pd.Series(portfolio_values, index=df_with_signals.index), trades

def quantile_loss(y_true, y_pred, quantile):
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def calculate_all_metrics(df, portfolio_values, trades, symbol, strategy_type):
    results = {}
    # --- Financial Metrics ---
    final_value = portfolio_values.iloc[-1]
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    returns = portfolio_values.pct_change().dropna()
    
    if len(df['timestamp']) > 1:
        time_diff = df['timestamp'].diff().mode()[0]
        periods_per_year = pd.Timedelta(days=365) / time_diff
    else: periods_per_year = 252
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else 0

    results["--- Financial Performance ---"] = ""
    results["Final Portfolio Value"] = f"${final_value:,.2f}"
    results["Total Return"] = f"{total_return:.2f}%"
    results["Sharpe Ratio (Annualized)"] = f"{sharpe_ratio:.2f}"
    
    completed_trades = [t for t in trades if 'exit_price' in t]
    if completed_trades:
        wins = sum(1 for t in completed_trades if t['profit_pct'] > 0)
        win_rate = (wins / len(completed_trades)) * 100
        results["Total Trades"] = len(completed_trades)
        results["Win Rate"] = f"{win_rate:.2f}%"
    else:
        results["Total Trades"] = 0
        results["Win Rate"] = "N/A"

    # --- Model-Specific Statistical Metrics ---
    if strategy_type == 'primary':
        df_eval = df.dropna(subset=['signal']).copy()
        df_eval['actual_movement'] = np.where(df_eval['close'].shift(-1) > df_eval['close'], 1, -1)
        
        results["\n--- Statistical Model Evaluation ---"] = ""
        if symbol in STABLE_ASSETS:
            active_signals_df = df_eval[df_eval['signal'] != 0]
            report = classification_report(
                active_signals_df['actual_movement'], active_signals_df['signal'], 
                labels=[-1, 1], target_names=['SELL Signal', 'BUY Signal'], zero_division=0
            )
            results["Classification Report"] = f"\n{report}"
        elif symbol in HIGH_VOLATILITY_ASSETS:
            y_true = df_eval['close'].shift(-1).dropna()
            y_pred_median = df_eval.loc[y_true.index, 'pred_median']
            r2 = r2_score(y_true, y_pred_median)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_median))
            q_loss_low = quantile_loss(y_true, df_eval.loc[y_true.index, 'pred_low'], 0.1)
            q_loss_high = quantile_loss(y_true, df_eval.loc[y_true.index, 'pred_high'], 0.9)
            
            results["R-Squared (Median Prediction)"] = f"{r2:.4f}"
            results["RMSE (Median Prediction)"] = f"${rmse:.8f}"
            results["Quantile Loss (Low 10%)"] = f"{q_loss_low:.8f}"
            results["Quantile Loss (High 90%)"] = f"{q_loss_high:.8f}"
            
    return results

# --- Main Execution ---
def main():
    assets_to_backtest = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS
    
    for symbol in assets_to_backtest:
        df, primary_model, chronos_model = load_data_and_models(symbol)
        if df is None: continue
            
        try:
            backtest_df = df.tail(2016).copy().reset_index(drop=True)
            
            # --- Pre-computation Step for Chronos ---
            chronos_pred_file = os.path.join(BASE_DATA_FOLDER, f"{symbol}_chronos_preds.csv")
            if os.path.exists(chronos_pred_file):
                print(f"Loading pre-computed Chronos predictions for {symbol}...")
                backtest_df['chronos_pred'] = pd.read_csv(chronos_pred_file, index_col=0, header=0).squeeze("columns")
            elif chronos_model:
                chronos_predictions = precompute_chronos_predictions(backtest_df, chronos_model)
                backtest_df['chronos_pred'] = chronos_predictions
                backtest_df['chronos_pred'].to_csv(chronos_pred_file)
                print(f"Saved Chronos predictions to {chronos_pred_file}")
            else:
                backtest_df['chronos_pred'] = np.nan

            # --- Run Backtest 1: Primary Model (KNN/LGBM) ---
            if primary_model:
                df1 = generate_primary_signals(backtest_df.copy(), primary_model, symbol)
                portfolio1, trades1 = run_backtest(df1)
                results1 = calculate_all_metrics(df1, portfolio1, trades1, symbol, 'primary')
                print(f"\n--- Backtest Results for {symbol} (Primary Model: {'KNN' if symbol in STABLE_ASSETS else 'LGBM'}) ---")
                for key, value in results1.items(): print(f"  {key}: {value}")
            
            # --- Run Backtest 2: Chronos T5 Model ---
            if not backtest_df['chronos_pred'].isna().all():
                df2 = generate_chronos_signals(backtest_df.copy())
                portfolio2, trades2 = run_backtest(df2)
                results2 = calculate_all_metrics(df2, portfolio2, trades2, symbol, 'chronos')
                print(f"\n--- Backtest Results for {symbol} (Chronos T5 Strategy) ---")
                for key, value in results2.items(): print(f"  {key}: {value}")

            # --- Run Backtest 3: Sentiment Model ---
            df3 = simulate_sentiment_signals(backtest_df.copy())
            portfolio3, trades3 = run_backtest(df3)
            results3 = calculate_all_metrics(df3, portfolio3, trades3, symbol, 'sentiment')
            print(f"\n--- Backtest Results for {symbol} (Sentiment Strategy) ---")
            for key, value in results3.items(): print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"\nAn error occurred during the backtest for {symbol}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
