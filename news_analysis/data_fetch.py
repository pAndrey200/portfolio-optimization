"""Candles from MOEX ISS + small helpers."""
import requests as rq
import pandas as pd

MOEX = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
    "securities/{t}/candles.json?from={s}&till={e}&interval=24"
)

def fetch_candles(ticker:str, start:str, end:str) -> pd.DataFrame:
    url = MOEX.format(t=ticker, s=start, e=end)
    j = rq.get(url, timeout=10).json()
    cols, data = j["candles"]["columns"], j["candles"]["data"]
    if not data:
        return pd.DataFrame(columns=["open","close","volume"])
    df = pd.DataFrame(data, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"])
    return df.set_index("begin")[["open","close","volume"]]