import os, time
from datetime import datetime, timedelta
from typing import Dict

import requests as rq
import pandas as pd

class LentaRuParser:
    BASE = "https://lenta.ru/search/v2/process?"

    def __init__(self, start:str, end:str, time_step:int=30, cache_dir:str="parsers/data"):
        self.start = datetime.strptime(start, "%Y-%m-%d")
        self.end   = datetime.strptime(end,   "%Y-%m-%d")
        self.step  = timedelta(days=time_step)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _build_url(p:Dict[str,str]) -> str:
        return (
            f"{LentaRuParser.BASE}from={p['from']}&size={p['size']}&sort={p['sort']}"
            f"&title_only={p['title_only']}&domain={p['domain']}&modified%2Cformat=yyyy-MM-dd"
            f"&type={p['type']}&bloc={p['bloc']}"
            f"&modified%2Cfrom={p['dateFrom']}&modified%2Cto={p['dateTo']}&query={p['query']}"
        )

    @staticmethod
    def _request(url:str) -> pd.DataFrame:
        time.sleep(1)  # API‑friendly
        r = rq.get(url, timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json()["matches"])

    def _load_cache(self, query:str) -> pd.DataFrame|None:
        fname = os.path.join(self.cache_dir, f"{query}_{self.start.date()}_{self.end.date()}.csv")
        if os.path.exists(fname):
            print(f"[Cache]  Using {fname}")
            return pd.read_csv(fname, parse_dates=["published"])
        return None

    def _save_cache(self, query:str, df:pd.DataFrame):
        fname = os.path.join(self.cache_dir, f"{query}_{self.start.date()}_{self.end.date()}.csv")
        df.to_csv(fname, index=False)
        print(f"[Cache]  Saved - {fname} | size - {len(df)}")

    def download(self, query_map:Dict[str,str]) -> pd.DataFrame:
        all_rows = []
        for q, tkr in query_map.items():
            cached = self._load_cache(q)
            if cached is not None:
                cached["ticker"] = tkr
                all_rows.append(cached)
                continue

            cur = self.start
            base = {
                "query": q, "from": "0", "size": "1000", "sort": "2",
                "title_only": "0", "type": "1", "bloc": "4", "domain": "1",
            }
            chunk_rows = []
            while cur <= self.end:
                nxt = min(cur + self.step, self.end)
                base.update(dateFrom=cur.strftime("%Y-%m-%d"), dateTo=nxt.strftime("%Y-%m-%d"))
                url = self._build_url(base)
                print(f"[Lenta] {q}  {base['dateFrom']} - {base['dateTo']}")
                try:
                    df = self._request(url)
                except Exception as e:
                    print("   !", e)
                    cur = nxt + timedelta(days=1)
                    continue
                if not df.empty:
                    df["ticker"] = tkr
                    df["text"] = df["text"].astype(str) + " " + df["snippet"].astype(str)
                    df["published"] = pd.to_datetime(df["pubdate"], unit="s")
                    df = df[["ticker", "published", "title", "text"]]
                    chunk_rows.append(df)
                cur = nxt + timedelta(days=1)

            if chunk_rows:
                df_q = pd.concat(chunk_rows, ignore_index=True)
                self._save_cache(q, df_q)
                all_rows.append(df_q)
        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()