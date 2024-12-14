import asyncio
from playwright.async_api import async_playwright
import pandas as pd

base_url = 'https://www.finam.ru'
ticket = 'ydex'
amount = 3

async def async_work():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False
        )
        page = await browser.new_page()
        await page.goto(base_url + '/quote/moex/' + ticket + '/publications/')
        div = await page.locator('div.mb2x').locator("a.cl-blue").all()
        s = []
        for d in div:
            s.append(await d.get_attribute("href"))
        urls = s[::2]
        
        news_data = []  # Список для хранения новостей
        i = 0
        for url in urls:
            if i == amount:
                break
            i += 1

            new_page = await browser.new_page()
            await new_page.goto(base_url + url)
            paragraphs = await new_page.locator('p').filter(
                has_not_text="Дизайн — «Липка и Друзья»"
            ).all()
            news_text = ""
            for paragraph in paragraphs:
                s = await paragraph.inner_text()
                if s == ('При полном или частичном использовании материалов ссылка на Finam.ru обязательна. Подробнее об использовании информации и котировок. Редакция не несет ответственности за достоверность информации, опубликованной в рекламных объявлениях.  18+'):
                    break
                news_text += s + "\n"

            news_data.append({"url": base_url + url, "content": news_text})
            print(i)
            await new_page.close()

        news_df = pd.DataFrame(news_data)
        news_df.to_csv(f"./data/{ticket}_news.csv", index=False, encoding='utf-8')

        await browser.close()

asyncio.run(async_work())
