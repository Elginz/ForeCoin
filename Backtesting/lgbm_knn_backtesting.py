"""
This script is used to evaluate and backtest the models, aligning with the objectives
set in the preliminary project report.

- For volatile coins (LGBM), it calculates RMSE, R-Squared, and Quantile Loss.
- For stable coins (KNN), it calculates a Classification Report (Precision, Recall, F1-Score).
- Profitability is assessed via Net Profit/Loss, Win Rate, and Sharpe Ratio.
"""

import pandas as pd
import numpy as np
import joblib
import os
import pandas_ta as ta
import glob
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from tqdm import tqdm

# --- Configuration ---
BASE_DATA_FOLDER = "historic_data"
MODELS_FOLDER = "trained_models"
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']

# --- Simulation Parameters ---
INITIAL_CAPITAL = 10000
TRADE_SIZE_PERCENT = 0.5
TRANSACTION_COST_PERCENT = 0.001

def find_asset_data_file(symbol):
    """Finds the latest data file for a given asset symbol."""
    subfolder = 'stable' if symbol in STABLE_ASSETS else 'volatile'
    search_path = os.path.join(BASE_DATA_FOLDER, subfolder, f"{symbol}_*.csv")
    files = glob.glob(search_path)
    if not files: return None
    return max(files, key=os.path.getmtime)

def load_data_and_model(symbol):
    """Loads historical data and the corresponding trained model."""
    print(f"\n--- Loading data and model for {symbol} ---")
    file_path = find_asset_data_file(symbol)
    if not file_path:
        print(f"  - Data file not found for {symbol}. Skipping.")
        return None, None
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    model_type = "knn_supertrend" if symbol in STABLE_ASSETS else "lgbm_quantile"
    model_path = os.path.join(MODELS_FOLDER, f"{symbol}_{model_type}_model.pkl")
    if not os.path.exists(model_path):
        print(f"  - Model file not found at '{model_path}'. Skipping.")
        return None, None
        
    model = joblib.load(model_path)
    print("  - Data and model loaded successfully.")
    return df, model

def generate_signals(df, model, symbol):
    """Generates trading signals based on the model and strategy for the asset type."""
    print(f"Generating signals for {symbol}...")
    df['signal'] = 0  # 1 for Buy, -1 for Sell, 0 for Hold

    if symbol in STABLE_ASSETS:
        # --- ENHANCED LOGIC for KNN + Supertrend Strategy ---
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
        supertrend_col = next((col for col in df.columns if col.startswith('SUPERT_')), None)
        if not supertrend_col:
            print("  - ERROR: Supertrend column not found.")
            return df
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', supertrend_col]
        X = df[feature_cols].dropna()
        if X.empty: return df

        df.loc[X.index, 'prediction'] = model.predict(X)
        
        # Two-part signal confirmation, as per project objectives
        price_predicts_increase = df['prediction'] > df['close']
        price_crosses_supertrend = df['close'] > df[supertrend_col]
        buy_signal = price_predicts_increase & price_crosses_supertrend
        
        price_predicts_decrease = df['prediction'] < df['close']
        price_crosses_below_supertrend = df['close'] < df[supertrend_col]
        sell_signal = price_predicts_decrease & price_crosses_below_supertrend
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1

    elif symbol in HIGH_VOLATILITY_ASSETS:
        # --- ENHANCED LOGIC for LGBM Quantile Strategy ---
        X = df[['open', 'high', 'low', 'close', 'volume']]
        df['pred_low'] = model['low'].predict(X)
        df['pred_median'] = model['median'].predict(X)
        df['pred_high'] = model['high'].predict(X)
        
        # Buy below the low quantile, sell above the high quantile
        df['signal'] = np.where(df['close'] < df['pred_low'], 1, 0)
        df['signal'] = np.where(df['close'] > df['pred_high'], -1, df['signal'])

    print("  - Signal generation complete.")
    return df

def run_backtest(df_with_signals):
    """Runs the backtesting simulation."""
    print("Running backtest simulation...")
    capital = INITIAL_CAPITAL
    position_size = 0
    position = 0  # 0 for no position, 1 for long
    portfolio_values = []
    trades = []

    for i in tqdm(range(len(df_with_signals)), desc="  - Simulating Trades"):
        row = df_with_signals.iloc[i]
        
        if row['signal'] == 1 and position == 0:
            trade_amount = capital * TRADE_SIZE_PERCENT
            position_size = trade_amount / row['close']
            capital -= position_size * row['close'] * (1 + TRANSACTION_COST_PERCENT)
            position = 1
            trades.append({'entry_price': row['close'], 'type': 'BUY'})
        
        elif row['signal'] == -1 and position == 1:
            capital += position_size * row['close'] * (1 - TRANSACTION_COST_PERCENT)
            position = 0
            position_size = 0
            # Finalize the last trade record
            if trades and 'exit_price' not in trades[-1]:
                last_trade = trades[-1]
                last_trade['exit_price'] = row['close']
                last_trade['profit_pct'] = (last_trade['exit_price'] - last_trade['entry_price']) / last_trade['entry_price']

        current_value = capital + (position_size * row['close'] if position == 1 else 0)
        portfolio_values.append(current_value)

    print("  - Simulation complete.")
    return pd.Series(portfolio_values, index=df_with_signals.index), trades

def quantile_loss(y_true, y_pred, quantile):
    """Calculates the quantile loss for evaluation."""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))

def calculate_metrics(df, portfolio_values, trades, symbol):
    """Calculates and returns all financial and statistical performance metrics."""
    results = {}
    
    # --- Financial Metrics ---
    final_value = portfolio_values.iloc[-1]
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    returns = portfolio_values.pct_change().dropna()
    
    # --- CORRECTED: Dynamically calculate annualization factor for Sharpe Ratio ---
    if len(df['timestamp']) > 1:
        time_diff = df['timestamp'].diff().mode()[0]
        periods_per_year = pd.Timedelta(days=365) / time_diff
    else:
        periods_per_year = 252 # Default to daily if not enough data
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
    df_eval = df.dropna(subset=['signal']).copy()
    df_eval['actual_movement'] = np.where(df_eval['close'].shift(-1) > df_eval['close'], 1, -1)
    
    results["\n--- Statistical Model Evaluation ---"] = ""
    if symbol in STABLE_ASSETS:
        # --- FIX: Explicitly define the labels to report on (ignore HOLD signals) ---
        # We only care about the performance of our active BUY (1) and SELL (-1) signals.
        active_signals_df = df_eval[df_eval['signal'] != 0]
        
        report = classification_report(
            active_signals_df['actual_movement'], 
            active_signals_df['signal'], 
            labels=[-1, 1], # Explicitly tell the report which labels to use
            target_names=['SELL Signal', 'BUY Signal'], 
            zero_division=0
        )
        results["Classification Report"] = f"\n{report}"
        
    elif symbol in HIGH_VOLATILITY_ASSETS:
        # Calculate regression and quantile metrics for LGBM
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


def main():
    """Main function to run the backtesting process."""
    assets_to_backtest = STABLE_ASSETS + HIGH_VOLATILITY_ASSETS
    
    for symbol in assets_to_backtest:
        df, model = load_data_and_model(symbol)
        if df is None or model is None: continue
            
        try:
            # Using 1 week of 5-min intervals as a consistent period (2016 data points)
            backtest_df = df.tail(2016).copy() 
            
            df_with_signals = generate_signals(backtest_df, model, symbol)
            portfolio_values, trades = run_backtest(df_with_signals)
            results = calculate_metrics(df_with_signals, portfolio_values, trades, symbol)
            
            print(f"\n--- Backtest Results for {symbol} (1-Week Period) ---")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"\nAn error occurred during the backtest for {symbol}: {e}")

if __name__ == "__main__":
    main()
