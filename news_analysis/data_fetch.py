import requests as rq
import pandas as pd
from datetime import datetime, timedelta

MOEX = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
    "securities/{t}/candles.json?from={s}&till={e}&interval=24"
)

def fetch_candles(ticker: str, start: str, end: str) -> pd.DataFrame:
    all_data = []
    cur_start = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    while cur_start <= end_date:
        url = MOEX.format(
            t=ticker,
            s=cur_start.strftime('%Y-%m-%d'),
            e=end_date.strftime('%Y-%m-%d')
        )
        try:
            resp = rq.get(url, timeout=10)
            resp.raise_for_status()
            j = resp.json()
        except Exception as e:
            print(f"Error fetching {ticker} from {cur_start}: {e}")
            break

        cols = j["candles"]["columns"]
        data = j["candles"]["data"]
        if not data:
            break

        df = pd.DataFrame(data, columns=cols)
        df["begin"] = pd.to_datetime(df["begin"])
        all_data.append(df)

        latest = df["begin"].max()
        cur_start = latest + timedelta(days=365)

        if len(df) < 2:
            break

    if not all_data:
        return pd.DataFrame(columns=["open", "close", "volume"])

    full_df = pd.concat(all_data).drop_duplicates("begin").set_index("begin")
    return full_df[["open", "close", "volume"]]