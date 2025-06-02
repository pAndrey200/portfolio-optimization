import asyncio
import os
from datetime import datetime
from playwright.async_api import async_playwright
import pandas as pd

class FinamParser:
    """
    Асинхронный парсер новостей с сайта Finam.ru.
    Возвращает DataFrame с колонками: ticker, published, title, text
    """
    BASE_URL = 'https://www.finam.ru'

    def __init__(self, ticket: str, max_articles: int = 1000, cache_dir: str = 'parsers/data'):
        self.ticket = ticket.upper()
        self.max_articles = max_articles
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        # кеш-файл для данного тикера
        self.cache_file = os.path.join(self.cache_dir, f"{self.ticket}_finam.csv")

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            print(f"[Finam][Cache] Using {self.cache_file}")
            return pd.read_csv(self.cache_file, parse_dates=['published'])
        return None

    def _save_cache(self, df: pd.DataFrame):
        df.to_csv(self.cache_file, index=False)
        print(f"[Finam][Cache] Saved → {self.cache_file}")

    async def _fetch(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            # Открываем страницу публикаций
            await page.goto(f"{self.BASE_URL}/quote/moex/{self.ticket}/publications/")
            print(self.ticket)
            # Кнопка загрузки дополнительных
            loaded = 0
            while loaded < self.max_articles:
                more = page.locator("[data-id='button-more']")
                if not await more.is_visible():
                    break
                await more.click()
                await page.wait_for_timeout(2000)
                loaded += 50

            # Собираем ссылки на новости
            links = await page.locator('div.mb2x a.cl-blue').all()
            urls = []
            for i,el in enumerate(links):
                href = await el.get_attribute('href')
                # берем каждую вторую ссылку (из двух рядом)
                if href and i % 2 == 0:
                    urls.append(self.BASE_URL + href)

            news = []
            for url in urls:
                print(url)
                try:
                    npg = await browser.new_page()
                    await npg.goto(url)
                    # заголовок
                    title = await npg.locator('h1').inner_text()
                    # дата публикации
                    date_raw = await npg.locator("[data-id='date']").inner_text()
                    published = datetime.strptime(date_raw, "%d.%m.%Y %H:%M")
                    # текст новости — все <p> до рекламного блока
                    paras = await npg.locator('p').all()
                    text = []
                    for p in paras:
                        txt = await p.inner_text()
                        if 'При полном или частичном использовании' in txt:
                            break
                        text.append(txt)
                    full_text = '\n'.join(text).strip()
                    news.append({
                        'ticker': self.ticket,
                        'published': published,
                        'title': title,
                        'text': full_text,
                        'url': url
                    })
                    await npg.close()
                except Exception as e:
                    print(f"[Finam][Error] {url}: {e}")

            await browser.close()
            return pd.DataFrame(news)

    def download(self) -> pd.DataFrame:
        # проверим кеш
        cached = self._load_cache()
        if cached is not None:
            return cached

        # иначе запускаем асинхронный парсинг
        df = asyncio.run(self._fetch())
        if not df.empty:
            self._save_cache(df)
        return df

# Пример использования:
# parser = FinamParser('CHMF', max_articles=500)
# finam_news = parser.download()
# print(finam_news.head())