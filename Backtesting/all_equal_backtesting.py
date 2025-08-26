"""
This script backtests a weighted ensemble strategy combining a primary model (KNN/LGBM),
Chronos T5, and Sentiment Analysis. Each component has a 1/3 weight in the final
trading decision.
"""

import pandas as pd
import numpy as np
import joblib
import os
import pandas_ta as ta
import glob
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import torch

# --- Import local python files ---
from predict_chronos import forecast_next_hour
from webscraper import determine_sentiment, BTCUSDT as BTC_QUERY, ETHUSDT as ETH_QUERY, BNBUSDT as BNB_QUERY, DOGEUSDT as DOGE_GUERY, SHIBUSDT as SHIB_QUERY

# --- Configuration ---
BASE_DATA_FOLDER = "historic_data"
MODELS_FOLDER = "trained_models"
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
ASSET_QUERIES = { 'BTCUSDT': BTC_QUERY, 'ETHUSDT': ETH_QUERY, 'BNBUSDT': BNB_QUERY, 'DOGEUSDT': DOGE_GUERY, 'SHIBUSDT': SHIB_QUERY }

# --- Simulation Parameters ---
INITIAL_CAPITAL = 10000
TRADE_SIZE_PERCENT = 0.5
TRANSACTION_COST_PERCENT = 0.001
CONVICTION_THRESHOLD = 0.5  # Threshold for the combined signal score to trigger a trade

def find_asset_data_file(symbol):
    subfolder = 'stable' if symbol in STABLE_ASSETS else 'volatile'
    search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
    files = glob.glob(search_path)
    if not files: return None
    return max(files, key=os.path.getmtime)

def load_data_and_models(symbol):
    print(f"\n--- Loading data and models for {symbol} ---")
    file_path = find_asset_data_file(symbol)
    if not file_path:
        print(f"  - Data file not found for {symbol}. Skipping.")
        return None, None, None
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Load primary model (KNN or LGBM)
    model_type = "knn_supertrend" if symbol in STABLE_ASSETS else "lgbm_quantile"
    model_path = os.path.join(MODELS_FOLDER, f"{symbol}_{model_type}_model.pkl")
    if not os.path.exists(model_path):
        print(f"  - Primary model file not found at '{model_path}'. Skipping.")
        return None, None, None
    primary_model = joblib.load(model_path)

    # Load Chronos model
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

def precompute_chronos_predictions(df, chronos_model):
    """Pre-computes Chronos predictions for the entire dataset to save time."""
    print("Pre-computing Chronos T5 predictions for the backtest period...")
    predictions = []
    pipeline = chronos_model['pipeline']
    context_size = 256 # A reasonable context window for predictions

    for i in tqdm(range(context_size, len(df)), desc="  - Forecasting with Chronos"):
        context = torch.tensor(df['close'].values[i-context_size:i], dtype=torch.float32).unsqueeze(0)
        forecast = pipeline.predict(context, 1)
        median_forecast = np.quantile(forecast[0].numpy(), 0.5, axis=0)[0]
        predictions.append(median_forecast)
    
    # Pad the start with NaN since we can't predict without enough history
    full_predictions = [np.nan] * context_size + predictions
    return pd.Series(full_predictions, index=df.index)

def simulate_sentiment(df, symbol):
    """Simulates sentiment analysis for historical data."""
    print(f"Simulating sentiment for {symbol}...")
    # This is a simplified simulation. In a real scenario, you'd use historical news data.
    # Here, we generate a random sentiment signal for demonstration purposes.
    # You can replace this with more sophisticated logic if you have historical sentiment data.
    sentiments = np.random.choice([-1, 0, 1], size=len(df), p=[0.25, 0.5, 0.25])
    return pd.Series(sentiments, index=df.index)

def generate_ensemble_signals(df, primary_model, symbol):
    """Generates a final trading signal based on the weighted average of all models."""
    print("Generating signals from all models...")

    # --- Signal 1: Primary Model (KNN or LGBM) ---
    df['primary_signal'] = 0
    if symbol in STABLE_ASSETS:
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
        supertrend_col = next((col for col in df.columns if col.startswith('SUPERT_')), None)
        if supertrend_col:
            feature_cols = ['open', 'high', 'low', 'close', 'volume', supertrend_col]
            X = df[feature_cols].dropna()
            if not X.empty:
                df.loc[X.index, 'prediction'] = primary_model.predict(X)
                buy = (df['prediction'] > df['close']) & (df['close'] > df[supertrend_col])
                sell = (df['prediction'] < df['close']) & (df['close'] < df[supertrend_col])
                df.loc[buy, 'primary_signal'] = 1
                df.loc[sell, 'primary_signal'] = -1
    elif symbol in HIGH_VOLATILITY_ASSETS:
        X = df[['open', 'high', 'low', 'close', 'volume']]
        df['pred_low'] = primary_model['low'].predict(X)
        df['pred_high'] = primary_model['high'].predict(X)
        df.loc[df['close'] < df['pred_low'], 'primary_signal'] = 1
        df.loc[df['close'] > df['pred_high'], 'primary_signal'] = -1
    
    # --- Signal 2: Chronos T5 ---
    df['chronos_signal'] = np.where(df['chronos_pred'] > df['close'], 1, -1)

    # --- Signal 3: Sentiment Analysis ---
    # This column was added by simulate_sentiment
    
    # --- Combine Signals with 1/3 Weighting ---
    df['conviction_score'] = (df['primary_signal'] * 0.333) + \
                             (df['chronos_signal'] * 0.333) + \
                             (df['sentiment_signal'] * 0.333)

    # --- Final Signal based on Conviction Threshold ---
    df['signal'] = 0
    df.loc[df['conviction_score'] > CONVICTION_THRESHOLD, 'signal'] = 1
    df.loc[df['conviction_score'] < -CONVICTION_THRESHOLD, 'signal'] = -1
    
    print("  - Ensemble signal generation complete.")
    return df

def run_backtest_and_get_results(df_with_signals):
    """Runs the simulation and calculates financial metrics."""
    print("Running backtest simulation...")
    capital = INITIAL_CAPITAL
    position_size = 0
    position = 0
    portfolio_values = []
    trades = []

    for i in tqdm(range(len(df_with_signals)), desc="  - Simulating Trades"):
        row = df_with_signals.iloc[i]
        
        if row['signal'] == 1 and position == 0:
            trade_amount = capital * TRADE_SIZE_PERCENT
            position_size = trade_amount / row['close']
            capital -= position_size * row['close'] * (1 + TRANSACTION_COST_PERCENT)
            position = 1
            trades.append({'entry_price': row['close']})
        
        elif row['signal'] == -1 and position == 1:
            capital += position_size * row['close'] * (1 - TRANSACTION_COST_PERCENT)
            position = 0
            position_size = 0
            if trades and 'exit_price' not in trades[-1]:
                last_trade = trades[-1]
                last_trade['exit_price'] = row['close']
                last_trade['profit_pct'] = (last_trade['exit_price'] - last_trade['entry_price']) / last_trade['entry_price']

        current_value = capital + (position_size * row['close'] if position == 1 else 0)
        portfolio_values.append(current_value)

    # --- Calculate Final Metrics ---
    portfolio_values = pd.Series(portfolio_values, index=df_with_signals.index)
    final_value = portfolio_values.iloc[-1]
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    returns = portfolio_values.pct_change().dropna()
    
    time_diff = df_with_signals['timestamp'].diff().mode()[0]
    periods_per_year = pd.Timedelta(days=365) / time_diff
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else 0
    
    completed_trades = [t for t in trades if 'exit_price' in t]
    wins = sum(1 for t in completed_trades if t['profit_pct'] > 0)
    win_rate = (wins / len(completed_trades)) * 100 if completed_trades else 0
    
    return {
        "--- Ensemble Strategy Performance ---": "",
        "Final Portfolio Value": f"${final_value:,.2f}",
        "Total Return": f"{total_return:.2f}%",
        "Sharpe Ratio (Annualized)": f"{sharpe_ratio:.2f}",
        "Total Trades": len(completed_trades),
        "Win Rate": f"{win_rate:.2f}%"
    }

def main():
    """Main function to orchestrate the entire backtesting process."""
    assets_to_backtest = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS
    
    for symbol in assets_to_backtest:
        df, primary_model, chronos_model = load_data_and_models(symbol)
        if df is None or primary_model is None: continue
            
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
                backtest_df['chronos_pred'] = backtest_df['close'] # Fallback if no chronos model

            # --- Simulate Sentiment ---
            backtest_df['sentiment_signal'] = simulate_sentiment(backtest_df, symbol)
            
            # --- Generate Final Signals and Run Backtest ---
            df_with_signals = generate_ensemble_signals(backtest_df, primary_model, symbol)
            results = run_backtest_and_get_results(df_with_signals)
            
            print(f"\n--- Backtest Results for {symbol} (Ensemble Strategy) ---")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"\nAn error occurred during the backtest for {symbol}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
