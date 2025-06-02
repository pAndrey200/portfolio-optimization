import os
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm

class RBCParser:
    BASE_SEARCH_URL = 'https://www.rbc.ru/search/ajax/'

    def __init__(self, project: str = 'quote', cache_dir: str = 'parsers/data'):
        self.project = project
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, query: str, date_from: str, date_to: str) -> str:
        filename = f"{query.upper()}_{date_from.replace('.', '-')}_{date_to.replace('.', '-')}.csv"
        return os.path.join(self.cache_dir, filename)

    def _load_cache(self, query: str, date_from: str, date_to: str) -> pd.DataFrame | None:
        path = self._cache_path(query, date_from, date_to)
        if os.path.exists(path):
            print(f"[RBC][Cache] Using {path}")
            return pd.read_csv(path, parse_dates=['published'])
        return None

    def _save_cache(self, df: pd.DataFrame, query: str, date_from: str, date_to: str):
        path = self._cache_path(query, date_from, date_to)
        df.to_csv(path, index=False)
        print(f"[RBC][Cache] Saved → {path}")

    def _build_search_url(self, params: dict) -> str:
        return (
            f"{self.BASE_SEARCH_URL}?project={params['project']}"
            f"&dateFrom={params['dateFrom']}&dateTo={params['dateTo']}"
            f"&query={params['query']}&page={params['page']}"
        )

    def _fetch_search_page(self, params: dict) -> pd.DataFrame:
        url = self._build_search_url(params)
        print(url)
        resp = rq.get(url)
        resp.raise_for_status()
        items = resp.json().get('items', [])
        return pd.DataFrame(items)

    def _iter_search(self, params: dict) -> pd.DataFrame:
        all_pages = []
        page = 1
        while True:
            params['page'] = str(page)
            df_page = self._fetch_search_page(params)
            if df_page.empty:
                break
            all_pages.append(df_page)
            page += 1
            if page == 90:
                page = 1
                last_date_str = df_page.tail(1)['publish_date'].values[0]
                last_date = pd.to_datetime(last_date_str, utc=True)
                new_date_to = last_date.strftime('%d.%m.%Y')
                params['dateTo'] = new_date_to
        if all_pages:
            return pd.concat(all_pages, ignore_index=True)
        return pd.DataFrame()

    def _get_article_text(self, url: str):
        resp = rq.get(url)
        resp.raise_for_status()
        soup = bs(resp.text, 'lxml')
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''
        paragraphs = soup.find_all('p')
        texts = []
        for p in paragraphs:
            txt = p.get_text(strip=True)
            if 'При полном или частичном использовании' in txt:
                break
            texts.append(txt)
        text = '\n'.join(texts)
        return title, text

    def get_articles(self, query: str, ticker: str, date_from: str, date_to: str) -> pd.DataFrame:

        cached = self._load_cache(query, date_from, date_to)
        if cached is not None:
            print(f"Ticker '{query.upper()}': найдено {len(cached)} статей (cache)")
            return cached

        params = {
            'project': self.project,
            'dateFrom': date_from,
            'dateTo': date_to,
            'query': query
        }
        df_search = self._iter_search(params)
        records = []
        for item in tqdm(df_search.to_dict('records'), desc=f"Parsing {query.upper()}", leave=False):
            url = item.get('fronturl')
            if not url:
                continue
            try:
                title, text = self._get_article_text(url)
                records.append({
                    'ticker': ticker,
                    'published': pd.to_datetime(item.get('publish_date'), utc=True).tz_convert('Europe/Moscow').tz_localize(None),
                    'title': title,
                    'text': text,
                    'url': url
                })
            except Exception:
                continue

        df = pd.DataFrame(records)
        # Сохраняем в кеш и выводим количество
        self._save_cache(df, query, date_from, date_to)
        print(f"Ticker '{query.upper()}': найдено {len(df)} статей")
        return df
