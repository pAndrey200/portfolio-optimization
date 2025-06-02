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
    def __init__(self, news:pd.DataFrame, prices:Dict[str,pd.DataFrame], window:int=3):
        self.tok = AutoTokenizer.from_pretrained("ai-forever/ruBERT-base")
        self.samples = []
        news['title'].fillna('', inplace=True)
        news['text'].fillna('', inplace=True)
        skipped = 0
        for _, r in news.iterrows():
            t = r.ticker
            date = r.published.normalize()
            if t not in prices or date not in prices[t].index:
                skipped += 1
                continue

            idx = prices[t].index.get_loc(date)
            if idx + window >= len(prices[t]):
                continue

            current_price = prices[t].iloc[idx].close
            future_price = prices[t].iloc[idx + window].close
            ret = future_price / current_price - 1
            label = int(ret > 0)
            full_text = remove_lines(f"{r.title} {r.text}")

            enc = self.tok(full_text,
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='pt')
            input_ids = enc.input_ids.squeeze(0)

            self.samples.append((
                input_ids,
                torch.tensor(label, dtype=torch.long),
            ))
        print(f"skipped {skipped}")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]