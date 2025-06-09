import argparse, torch
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, random_split
from transformers import logging as hf_logging
from tqdm.auto import tqdm
from parsers.lenta_parser import LentaRuParser
from parsers.rbc_parser import RBCParser
from data_fetch import fetch_candles
from dataset import FusionDataset
from model import FusionModel
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

DEFAULT_MAP = {
    "сбербанк":       "SBER",
    "лукойл":         "LKOH",
    "роснефть":       "ROSN",
    "новатэк":        "NVTK",
    "газпром":        "GAZP",
    "полюс":          "PLZL",
    "норникель":      "GMKN",
    "татнефть":       "TATN",
    "фосагро":        "PHOR",
    "северсталь":     "CHMF",
}

def save_dataset(news: pd.DataFrame, dataset: FusionDataset, path='fusion_dataset.csv'):
    records = []
    for i, (_, label) in enumerate(dataset):
        r = news.iloc[i]
        records.append({
            'ticker': r.ticker,
            'published': r.published,
            'title': r.title,
            'text': r.text,
            'label': label.item(),
        })
    pd.DataFrame(records).to_csv(path, index=False)




def parse_query_map(s:str):
    if not s: return DEFAULT_MAP
    d = {}
    for p in s.split(','):
        if ':' in p:
            q,t = p.split(':',1)
            d[q.strip()] = t.strip().upper()
    return d or DEFAULT_MAP

hf_logging.set_verbosity_error()

CE = torch.nn.CrossEntropyLoss()

def run_epoch(model, loader, optim, device, train=True):
    model.train() if train else model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    batches = 0

    loop = tqdm(loader, desc='Train' if train else 'Val', leave=False)
    for input_ids, labels in loop:
        input_ids, labels = input_ids.to(device), labels.to(device)

        if train:
            optim.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(input_ids)
            loss = CE(outputs, labels)
            if train:
                loss.backward()
                optim.step()

        total_loss += loss.item()
        batches += 1

        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    avg_loss = total_loss / batches

    if train:
        print(f"[Train] Loss: {avg_loss:.4f}  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
    else:
        print(f"[Val]   Loss: {avg_loss:.4f}  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    return {
        'loss':   avg_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--queries", default="")
    ap.add_argument("--horizon", type=int, default=5,
                    help="горизонт в торговых днях для label")
    ap.add_argument("--pad", type=int, default=365,
                    help="сколько дней истории добрать ДО start")
    ap.add_argument("--label-type", choices=["point", "mean"],
                    default="point")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    qmap = parse_query_map(args.queries)

    # ---------- парсим новости ----------
    news_lenta = LentaRuParser(args.start, args.end).download(qmap)

    parser_rbc = RBCParser()
    rbc_frames = []
    for q, tkr in qmap.items():
        df = parser_rbc.get_articles(
            q, tkr,
            datetime.strptime(args.start, "%Y-%m-%d").strftime("%d.%m.%Y"),
            datetime.strptime(args.end,   "%Y-%m-%d").strftime("%d.%m.%Y"),
        )
        if not df.empty:
            rbc_frames.append(df)

    news = (pd.concat([news_lenta, *rbc_frames], ignore_index=True)
              .sort_values("published"))

    # ---------- качаем цены ----------
    pad_start = (pd.to_datetime(args.start) - timedelta(days=args.pad)).strftime("%Y-%m-%d")
    prices = {t: fetch_candles(t, pad_start, args.end) for t in set(qmap.values())}

    # ---------- готовим датасет ----------
    ds = FusionDataset(news, prices,
                       horizon=args.horizon,
                       label_type=args.label_type)
    save_dataset(news, ds, "fusion_dataset.csv")
    # train,val = random_split(ds, [int(.8*len(ds)), len(ds)-int(.8*len(ds))])
    # ld_tr = DataLoader(train,batch_size=8,shuffle=True)
    # ld_val= DataLoader(val,batch_size=8)

    # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(dev)
    # model = FusionModel().to(dev)
    # optim = torch.optim.AdamW(model.parameters(),lr=3e-5, weight_decay=1e-2)

    # for e in range(1,args.epochs+1):
    #     train_stats = run_epoch(model, ld_tr, optim, dev, True)
    #     val_stats   = run_epoch(model, ld_val, optim, dev, False)
    # torch.save(model.state_dict(),'fusion_model.pt')

if __name__=='__main__':
    main()