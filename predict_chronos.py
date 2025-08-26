"""
This file is used to predict the price changes using the Chronos T5 time series forecasting model. 

It should work simultaneously with the other models.

It takes in a file input, which is the path of the file, and returns the output via terminal

"""

from chronos import ChronosPipeline
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import set_seed

# These paths are typically used for initial data loading in a script's main block
# In a web application, these are usually managed by the main app.py file
BTCUSDT = './historic_data/stable/BTCUSDT_data.csv'
BNBUSDT = './historic_data/stable/BNBUSDT_data.csv'
ETHUSDT = './historic_data/stable/ETHUSDT_data.csv'
DOGEUSDT = './historic_data/volatile/DOGEUSDT_data.csv'
SHIBUSDT = './historic_data/volatile/SHIBUSDT_data.csv'


def load_data(file_path):
    """
    Loads historical data from a CSV, prepares values and initial context for Chronos,
    and initializes the Chronos pipeline.
    Returns the DataFrame, values, context tensor, and pipeline.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    values = df["close"].values
    # Context needs to be unsqueezed to add a batch dimension for pipeline.predict
    context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # shape [1, time_steps]

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    return df, values, context, pipeline


def forecast_next_hour(df, values, context, pipeline, prediction_length=1):
    """
    Predicts the next hour's price using the Chronos T5 model.
    
    Args:
        df (pd.DataFrame): The historical DataFrame, needed to determine the frequency.
        values (np.array): The historical close prices.
        context (torch.Tensor): The context tensor for the Chronos pipeline.
        pipeline (ChronosPipeline): The loaded Chronos pipeline.
        prediction_length (int): The number of future steps to predict.
        
    Returns:
        tuple: A tuple containing forecast_timestamps, median, low, high.
    """
    # Reproducibility
    set_seed(42)

    # Forecast
    # The context already has the batch dimension from load_data or previous updates
    forecast = pipeline.predict(context, prediction_length)  # shape: [1, num_samples, prediction_length]
    forecast_np = forecast[0].numpy()

    # Get quantiles
    low, median, high = np.quantile(forecast_np, [0.1, 0.5, 0.9], axis=0)

    # Determine forecast timestamps
    # This now correctly uses the 'df' passed as an argument
    last_timestamp = df["timestamp"].iloc[-1]
    
    # Calculate frequency dynamically
    # Ensure there are enough data points to calculate a meaningful frequency
    if len(df["timestamp"]) > 1:
        freq = df["timestamp"].diff().mode()[0]
    else:
        # Default to hourly if not enough data to infer frequency
        freq = pd.Timedelta(hours=1) 
        
    forecast_timestamps = pd.date_range(start=last_timestamp + freq, periods=prediction_length, freq=freq)

    # Decision logic for next hour (moved to app.py for display there)
    # last_price = values[-1]
    # next_hour_prediction = median[0]

    return forecast_timestamps, median, low, high


def plot_forecast(df, values, forecast_timestamps, median, low, high, lookback=100):
    """
    Plots the historical data and the Chronos forecast.
    """
    plt.figure(figsize=(10, 5))
    
    historical_timestamps = df["timestamp"].iloc[-lookback:]
    plt.plot(historical_timestamps, values[-lookback:], label="Historical", color="blue")

    plt.plot(forecast_timestamps, median, label="Forecast (median)", color="red")
    plt.fill_between(forecast_timestamps, low, high, color="orange", alpha=0.3, label="80% confidence")

    plt.title("Chronos-T5 Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage when running predict_chronos.py directly
    df, values, context, pipeline = load_data(ETHUSDT)
    # Pass df to forecast_next_hour when calling it directly
    forecast_index, median, low, high = forecast_next_hour(df, values, context, pipeline)
    
    # Calculate last_price and next_hour_prediction here for standalone script output
    last_price = values[-1]
    next_hour_prediction = median[0]

    print(f"Last price: {last_price:.2f}")
    print(f"Predicted next hour price (median): {next_hour_prediction:.2f}")
    
    plot_forecast(df, values, forecast_index, median, low, high)
