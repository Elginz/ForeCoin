"""
This file contains helper functions used to collect market data.

It should run taking an input of START and END date

It should also take in an input of the HIGH_VOLATILITY_ASSETS and STABLE_ASSETS
"""

from binance.client import Client
import pandas as pd
from datetime import datetime
import os


#  Helper Function to get the correct time interval 
def get_api_interval(interval_str):
    # Dictionary to map strings to the API constants
    intervals = {
        "1min": Client.KLINE_INTERVAL_1MINUTE,
        "5min": Client.KLINE_INTERVAL_5MINUTE,
        "1hr": Client.KLINE_INTERVAL_1HOUR,
        "12hr": Client.KLINE_INTERVAL_12HOUR,
        "1day": Client.KLINE_INTERVAL_1DAY
    }
    # Get the constant from dictionary 
    interval = intervals.get(interval_str)
    if interval is None:
        # Error handler if not supported
        raise ValueError("Unsupported time interval: choose from '1min', '5min', '1hr', '12hr', '1day'")
    return interval

# Function to save data into same CSV file  
def save_or_update_csv(new_df, symbol, output_folder):
    # Path for output file 
    path = os.path.join(output_folder, f"{symbol}_data.csv")
    # Check if CSV file exists
    if os.path.exists(path):
        # If exist, read old data from CSV into df and convert columns to datetime
        existing_df = pd.read_csv(path)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        # Combine the old and new df
        combined = pd.concat([existing_df, new_df])
        # Sort data by timestamp
        combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp')
    else:
        # If file doesnt exist, new data
        combined = new_df
    combined.to_csv(path, index=False)


#  Function to Fetch and Save Data 
def fetch_and_save_data(symbols, start_date_str, end_date_str, output_folder, interval_str="1hr", api_key=None, api_secret=None):
    """
        OHLCV data for the following
    
        symbols (list): list of trading symbols (e.g., ['BTCUSDT']).
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
        output_folder (str): The path to the folder where CSV files will be saved.
        interval_str (str): The candle interval (e.g., '1hr').
    """
    # Initialise Binance client
    client = Client(api_key, api_secret)
    api_interval = get_api_interval(interval_str)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    for symbol in symbols:
        print(f"\nFetching {interval_str} data for {symbol} from {start_date_str} to {end_date_str}...")
        try:
            # Fetch historical klines (candlestick)
            klines = client.get_historical_klines(
                symbol,
                api_interval,
                start_date_str,
                end_str=end_date_str
            )
            # Error handler if API returns no data
            if not klines:
                print(f"No data found for {symbol} in the given range.")
                continue
            # Convert into pandas df to process
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convert timestamp to datetime and process numeric columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Ensure financial columns are treated as numbers
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only the essential columns for the models
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            print(f"Successfully fetched {len(df)} records for {symbol}.")

            # Save to CSV with the 'timestamp' column, not as an index.
            save_or_update_csv(df, symbol, output_folder)  
            print(f"Data for {symbol} updated")

        # Error handler
        except Exception as e:
            print(f"An error occurred while fetching data for {symbol}: {e}")

#  Main Function. calls data fetching for stbale and volatile assets
def run_data_gathering_process(start_date, end_date, stable_assets, high_volatility_assets):
    # If end_date is in the future, the API will fetch data up to the latest available.
    base_folder = "historic_data"
    stable_folder = os.path.join(base_folder, "stable")
    volatile_folder = os.path.join(base_folder, "volatile")
    
    print("--- Starting Data Gathering Process ---")

    # Fetch and save data for STABLE assets
    print("\n" + "="*50)
    print("Processing STABLE assets...")
    print("="*50)
    fetch_and_save_data(stable_assets, start_date, end_date, stable_folder, interval_str="1hr")

    # Fetch and save data for HIGH VOLATILITY assets
    print("\n" + "="*50)
    print("Processing HIGH VOLATILITY assets...")
    print("="*50)
    fetch_and_save_data(high_volatility_assets, start_date, end_date, volatile_folder, interval_str="5min")
    
    print("\n---  Data Gathering Process Complete ---")
