"""FusionDataset: (tokenized text, price window) → label"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from typing import Dict
import re

def remove_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Низкие комиссии. Ежедневная подборка инвестиционных идей":
            continue
        if re.fullmatch(r"[\-,]+", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


class FusionDataset(Dataset):
    """
    label_type = "point" : (Close_{t+h} / Close_{t-1}) - 1
    label_type = "mean"  : (MeanClose_{t+1:t+h} / Close_{t-1}) - 1
    """
    def __init__(self, news: pd.DataFrame,
                 prices: Dict[str, pd.DataFrame],
                 horizon: int = 5,
                 label_type: str = "point",
                 max_len: int = 128):
        self.tok = AutoTokenizer.from_pretrained("ai-forever/ruBERT-base")
        self.samples = []

        news = news.copy()
        news["title"].fillna("", inplace=True)
        news["text"].fillna("", inplace=True)

        skipped = 0
        for _, row in news.iterrows():
            tkr = row.ticker
            if tkr not in prices:
                skipped += 1
                continue

            pub_day = row.published.normalize()

            price_idx = prices[tkr].index    # DatetimeIndex
            idx_today = price_idx.searchsorted(pub_day, side="left")
            if idx_today == len(price_idx):  # после последней свечи
                skipped += 1
                continue

            if idx_today == 0 or idx_today + horizon >= len(price_idx):
                skipped += 1
                continue
            idx_prev = idx_today - 1

            close_prev = prices[tkr].iloc[idx_prev].close
            if label_type == "point":
                close_future = prices[tkr].iloc[idx_prev + horizon].close
            elif label_type == "mean":
                close_future = prices[tkr].iloc[idx_prev+1 : idx_prev+horizon+1].close.mean()
            else:
                raise ValueError("label_type must be 'point' or 'mean'")

            ret = close_future / close_prev - 1.0
            label = int(ret > 0)          # 0 = падение/нейтр., 1 = рост

            full_text = remove_lines(f"{row.title} {row.text}")
            enc = self.tok(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )

            self.samples.append((
                enc.input_ids.squeeze(0),
                torch.tensor(label, dtype=torch.long)
            ))

        print(f"FusionDataset: {len(self.samples)} samples, skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]