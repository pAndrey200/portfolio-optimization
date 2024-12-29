
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class lentaRu_parser:
    def __init__(self):
        pass
    
    def _get_url(self, param_dict: dict) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями

        url = 'https://lenta.ru/search/v2/process?'\
        + 'from=0&'\                       # Смещение
        + 'size=1000&'\                    # Кол-во статей
        + 'sort=2&'\                       # Сортировка по дате (2), по релевантности (1)
        + 'title_only=0&'\                 # Точная фраза в заголовке
        + 'domain=1&'\                     # ??
        + 'modified%2Cformat=yyyy-MM-dd&'\ # Формат даты
        + 'type=1&'\                       # Материалы. Все материалы (0). Новость (1)
        + 'bloc=4&'\                       # Рубрика. Экономика (4). Все рубрики (0)
        + 'modified%2Cfrom=2020-01-01&'\
        + 'modified%2Cto=2020-11-01&'\
        + 'query='                         # Поисковой запрос
        """
        hasType = int(param_dict['type']) != 0
        hasBloc = int(param_dict['bloc']) != 0
        url = 'https://lenta.ru/search/v2/process?'\
        + 'from={}&'.format(param_dict['from'])\
        + 'size={}&'.format(param_dict['size'])\
        + 'sort={}&'.format(param_dict['sort'])\
        + 'title_only={}&'.format(param_dict['title_only'])\
        + 'domain={}&'.format(param_dict['domain'])\
        + 'modified%2Cformat=yyyy-MM-dd&'\
        + 'type={}&'.format(param_dict['type']) * hasType\
        + 'bloc={}&'.format(param_dict['bloc']) * hasBloc\
        + 'modified%2Cfrom={}&'.format(param_dict['dateFrom'])\
        + 'modified%2Cto={}&'.format(param_dict['dateTo'])\
        + 'query={}'.format(param_dict['query'])
        
        return url


    def _get_search_table(self, param_dict: dict, ticket:str) -> pd.DataFrame:
        """
        Возвращает pd.DataFrame со списком статей
        """
        time.sleep(2)
        url = self._get_url(param_dict)
        r = rq.get(url)
        search_table = pd.DataFrame(r.json()['matches'])
        search_table['ticket'] = ticket
        self._get_stock(search_table, ticket)
        return search_table
    
    def _get_stock(self, df: pd.DataFrame, ticket:str):
        """
        Добавляет разницу цен акций на момент начала и конца торгов
        """
        for index, row in df.iterrows():
            ticket = ('YNDX' if row['pubdate'] < 1719532800 else 'YDEX') if ticket == 'YANDEX' else ticket
            j = rq.get('http://iss.moex.com/iss/engines/stock/markets/shares/securities/' + ticket + '/candles.json?from=' + datetime.fromtimestamp(row['pubdate']).strftime('%Y-%m-%d') + '&till=' + datetime.fromtimestamp(row['pubdate'] + 86400*3).strftime('%Y-%m-%d') + '&interval=24').json()
            stock = [{k : r[i] for i, k in enumerate(j['candles']['columns'])} for r in j['candles']['data']]
            print('http://iss.moex.com/iss/engines/stock/markets/shares/securities/' + ticket + '/candles.json?from=' + datetime.fromtimestamp(row['pubdate']).strftime('%Y-%m-%d') + '&till=' + datetime.fromtimestamp(row['pubdate'] + 86400*3).strftime('%Y-%m-%d') + '&interval=24', stock)
            df.at[index, 'target'] = (stock[0]['close'] - stock[0]['open']) / stock[0]['open'] if stock != [] else None
    
    def _get_articles(self,
                     param_dict,
                     time_step,
                     ticket,
                     save_excel = True) -> pd.DataFrame:
        """
        Функция для скачивания статей интервалами через каждые time_step дней

        param_dict: dict
        ### Параметры запроса 
        ###### project - раздел поиска, например, rbcnews
        ###### category - категория поиска, например, TopRbcRu_economics
        ###### dateFrom - с даты
        ###### dateTo - по дату
        ###### offset - смещение поисковой выдачи
        ###### limit - лимит статей, максимум 100
        ###### query - поисковой запрос (ключевое слово), например, РБК

        """
        param_copy = param_dict.copy()
        time_step = timedelta(days=time_step)
        dateFrom = datetime.strptime(param_copy['dateFrom'], '%Y-%m-%d')
        dateTo = datetime.strptime(param_copy['dateTo'], '%Y-%m-%d')
        if dateFrom > dateTo:
            raise ValueError('dateFrom should be less than dateTo')
        
        out = pd.DataFrame()
        while dateFrom <= dateTo:
            param_copy['dateTo'] = (dateFrom + time_step).strftime('%Y-%m-%d')
            if dateFrom + time_step > dateTo:
                param_copy['dateTo'] = dateTo.strftime('%Y-%m-%d')
            print('Parsing articles from '\
                  + param_copy['dateFrom'] +  ' to ' + param_copy['dateTo'])
            out = pd.concat([out, self._get_search_table(param_copy, ticket)], axis=0, ignore_index=True)
            dateFrom += time_step + timedelta(days=1)
            param_copy['dateFrom'] = dateFrom.strftime('%Y-%m-%d')
        
        print('Finish')
        return out
    
    def get_dataset(self, tickets):



        df = pd.DataFrame()
        for query, ticket in tickets.items():
            time.sleep(3)
            offset = 0
            size = 1000
            sort = "2"
            title_only = ""
            domain = "1"
            material = "0"
            bloc = "4"
            dateFrom = '2024-01-01'
            dateTo = "2024-12-24"
            param_dict = {'query'     : query, 
                'from'      : str(offset),
                'size'      : str(size),
                'dateFrom'  : dateFrom,
                'dateTo'    : dateTo,
                'sort'      : sort,
                'title_only': title_only,
                'type'      : material, 
                'bloc'      : bloc,
                'domain'    : domain}
            new_df = self._get_articles(param_dict=param_dict,
                            time_step = 30,
                            ticket = ticket,
                            save_excel = True)
            new_df.to_csv("./data/lenta_{}_{}_{}.csv".format(param_dict['dateFrom'], param_dict['dateTo'], ticket))
            df = pd.concat([df, new_df], axis=0, ignore_index=True)
            
        df.to_excel("./data/lenta_{}_{}.xlsx".format(
        param_dict['dateFrom'],
        param_dict['dateTo']))
        df.to_csv("./data/lenta_{}_{}.csv".format(
        param_dict['dateFrom'],
        param_dict['dateTo']))
        return df
         



tickets = {
    'яндекс' : 'YANDEX',
    'сбер' : 'SBER',
    'тинькофф' : 'T',
    'газпром' : 'GAZP',
    'лукойл' : 'LKOH',
    'норникель' : 'GMKN',
    'роснефть' : 'ROSN',
    'втб' : 'VTBR',
    'московская биржа' : 'MOEX'
}
lp = lentaRu_parser()
df = lp.get_dataset(tickets)
