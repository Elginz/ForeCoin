# live_data.py

"""
This file serves to continunously collect data on the respective assets.

It uses the data_collect.py helper functions
"""
import schedule
import time
import pytz
from datetime import datetime
import threading
import data_collect

# Assets to collect data for
HIGH_VOLATILITY_ASSETS = ['DOGEUSDT', 'SHIBUSDT']
STABLE_ASSETS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

# this needs to be changed, to update everyday
# YYMMDD
START_DATE = "2025-08-01"

def update_and_run_data_collect():
    est = pytz.timezone('US/Eastern')
    now_est = datetime.now(est)
    end_date = now_est.strftime('%Y-%m-%d')

    print(f"\n[Scheduler] Running data collection with END_DATE = {end_date}")
    try:
        data_collect.run_data_gathering_process(START_DATE,
                                                end_date,
                                                STABLE_ASSETS,
                                                HIGH_VOLATILITY_ASSETS)
        
        print("[Scheduler] Data collection completed successfully.")
    except Exception as e:
        print(f"[Scheduler Error] {e}")

def run_scheduler():

    # This is for the schedule to run every hour, exactly on the hour and 30 minutes
    schedule.every().hour.at(":30").do(update_and_run_data_collect)

    # This is for the schedule to run every 12:00 
    # schedule.every().day.at("12:00").do(update_and_run_data_collect)

    # FOR TESTING ONLY
    # schedule.every(1).minutes.do(update_and_run_data_collect)

    print("[Scheduler] Waiting for scheduled jobs...")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()

