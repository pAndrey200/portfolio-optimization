import asyncio
from playwright.async_api import async_playwright, expect
import os 

base_url = 'https://www.finam.ru'
ticket = 'ydex'
amount = 10

async def async_work():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel='chrome',
            headless=False
        )
        page = await browser.new_page()
        await page.goto(base_url + '/quote/moex/' + ticket + '/publications/')
        div = await page.locator('div.mb2x').locator("a.cl-blue").all()
        s = []
        for d in div:
            s.append(await d.get_attribute("href"))
        urls = s[::2]
        if not os.path.exists("./data/" + ticket): 
            os.makedirs("./data/" + ticket)
        i = 0
        for url in urls:
            if i == amount:
                break
            i += 1
            if(os.path.exists("./data/" + ticket + url[url.find('item/') + 4:-1] + '.txt')):
                continue
            new_page = await browser.new_page()
            await page.goto(base_url + url)
            paragraphs = await page.locator('p').filter(has_not_text="Дизайн — «Липка и Друзья»").all()
            with open('./data/' + ticket  + url[url.find('item/') + 4:-1] + '.txt', 'a', encoding='utf-8') as f:
                for paragraph in paragraphs:
                    s = await paragraph.inner_text()
                    if (s == ('При полном или частичном использовании материалов ссылка на\xa0Finam.ru\xa0обязательна. Подробнее об\xa0использовании информации и\xa0котировок. Редакция не\xa0несет ответственности за\xa0достоверность информации, опубликованной в\xa0рекламных объявлениях.\xa0\xa018+')):
                        break
                    f.write(s)
        await browser.close()

asyncio.run(async_work())