import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import requests
import matplotlib.pyplot as plt

NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

def get_financial_news(query, api_key, max_results=3):
    financial_domains = "bloomberg.com,cnbc.com,reuters.com,wsj.com,marketwatch.com,ft.com,yahoo.com,forbes.com,investopedia.com"

    url = (f"https://newsapi.org/v2/everything?"
           f"q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&domains={financial_domains}")

    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])[:max_results]
        return articles
    else:
        return []


popular_tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "ASML", "NVDA", "QCOM", "QUBT"]

st.title("AI24: Project, team 31")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

st.header("Выберите тикер актива")

cols = st.columns(len(popular_tickers))
for i, ticker in enumerate(popular_tickers):
    if cols[i].button(ticker):
        st.session_state.selected_ticker = ticker

user_ticker = st.text_input("Или введите свой тикер:", "")

if user_ticker:
    st.session_state.selected_ticker = user_ticker

if st.session_state.selected_ticker:
    selected_ticker = st.session_state.selected_ticker
    try:
        stock = yf.Ticker(selected_ticker)

        st.header(f"Основная информация: {selected_ticker}")
        info = stock.info
        st.write(f"**Название:** {info.get('longName', 'Неизвестно')}")
        st.write(f"**Сектор:** {info.get('sector', 'Неизвестно')}")
        st.write(f"**Биржа:** {info.get('exchange', 'Неизвестно')}")

        market_cap = info.get('marketCap', None)
        if market_cap:
            market_cap_bln = market_cap / 1e9
            st.write(f"**Оценочная стоимость:** {market_cap_bln:.2f} млрд долларов")
        else:
            st.write("**Рыночная капитализация:** Неизвестно")

        price = info.get('regularMarketPrice', None)
        if not price:
            data = stock.history(period="1d", interval="1d")
            if not data.empty:
                price = data['Close'].iloc[-1]
        if price:
            st.write(f"**Текущая цена:** ${price:.2f}")
        else:
            st.write("**Текущая цена:** Неизвестно")

        st.header("График цены")
        period = st.selectbox("Выберите период:", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=2)
        interval = st.selectbox("Выберите интервал:", ["1d", "1wk", "1mo"], index=0)
        data = stock.history(period=period, interval=interval)

        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Цена закрытия'))
            fig.update_layout(
                title=f'График цены для {selected_ticker}',
                xaxis_title="Дата",
                yaxis_title="Цена закрытия",
                xaxis_rangeslider_visible=True,
                template="plotly_dark"
            )
            st.plotly_chart(fig)

            st.header("Финансовые новости")
            st.write(f"Последние новости о {selected_ticker}:")
            news = get_financial_news(selected_ticker, NEWS_API_KEY)

            if news:
                for article in news:
                    st.write(f"**[{article['title']}]({article['url']})**")
                    st.write(f"*{article['source']['name']}* | Опубликовано: {article['publishedAt']}")
                    st.write(f"Описание: {article['description']}")
            else:
                st.write("Нет доступных новостей.")

            st.header("Ключевые численные показатели")
            metrics_columns = st.columns(3)
            metrics_columns[0].metric(label="Средняя цена закрытия", value=f"{data['Close'].mean():.2f}")
            metrics_columns[1].metric(label="Медианная цена закрытия", value=f"{data['Close'].median():.2f}")
            metrics_columns[2].metric(label="Максимальная цена закрытия", value=f"{data['Close'].max():.2f}")
            metrics_columns = st.columns(3)
            metrics_columns[0].metric(label="Минимальная цена закрытия", value=f"{data['Close'].min():.2f}")
            metrics_columns[1].metric(label="Изменение (с начальной до конечной)",
                                      value=f"{((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100:.2f}%")
            metrics_columns[2].metric(label="Среднее дневное изменение (%)",
                                      value=f"{data['Close'].pct_change().mean() * 100:.2f}%")
            metrics_columns = st.columns(3)
            metrics_columns[0].metric(label="Средний объем торгов", value=f"{data['Volume'].mean():.2f}")
            metrics_columns[1].metric(label="Максимальный объем торгов", value=f"{data['Volume'].max():.2f}")
            metrics_columns[2].metric(label="Минимальный объем торгов", value=f"{data['Volume'].min():.2f}")

            data['Daily Return'] = data['Close'].pct_change()
            volatility = np.std(data['Daily Return']) * np.sqrt(len(data))
            st.write(f"**Годовая волатильность:** {volatility:.2%}")

            candle = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlesticks"
            )])
            candle.update_layout(
                title=f'Японские свечи для {selected_ticker}',
                xaxis_title="Дата",
                yaxis_title="Цена",
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            st.plotly_chart(candle)

            data['Bollinger_Mid'] = data['Close'].rolling(window=20).mean()
            data['Bollinger_Up'] = data['Bollinger_Mid'] + (data['Close'].rolling(window=20).std() * 2)
            data['Bollinger_Low'] = data['Bollinger_Mid'] - (data['Close'].rolling(window=20).std() * 2)

            fig_bollinger = go.Figure()
            fig_bollinger.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Цена закрытия'))
            fig_bollinger.add_trace(
                go.Scatter(x=data.index, y=data['Bollinger_Up'], mode='lines', name='Bollinger Верхний',
                           line=dict(dash='dot')))
            fig_bollinger.add_trace(
                go.Scatter(x=data.index, y=data['Bollinger_Low'], mode='lines', name='Bollinger Нижний',
                           line=dict(dash='dot')))
            fig_bollinger.update_layout(
                title=f'Bollinger bands для {selected_ticker}',
                xaxis_title="Дата",
                yaxis_title="Цена закрытия",
                template="plotly_dark"
            )
            st.plotly_chart(fig_bollinger)

            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            data = data.dropna()

            if len(data) < 60:
                st.warning(
                    "Для выполнения декомпозиции временного ряда необходимо как минимум 60 точек данных. Попробуйте выбрать более длительный период.")
            else:
                result = seasonal_decompose(data['Close'], model='additive', period=30)

                fig_decompose_1 = go.Figure()
                fig_decompose_1.add_trace(
                    go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Оригинальный ряд',
                               line=dict(color='royalblue')))
                fig_decompose_1.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Тренд',
                                                     line=dict(color='orange')))
                fig_decompose_1.update_layout(
                    title=f'Оригинальный ряд и Тренд для {selected_ticker}',
                    xaxis_title="Дата",
                    yaxis_title="Значение",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_decompose_1)

                fig_decompose_2 = go.Figure()
                fig_decompose_2.add_trace(
                    go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Сезонность',
                               line=dict(color='green')))
                fig_decompose_2.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Остатки',
                                                     line=dict(color='red')))
                fig_decompose_2.update_layout(
                    title=f'Сезонность и Остатки для {selected_ticker}',
                    xaxis_title="Дата",
                    yaxis_title="Значение",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_decompose_2)

        st.header("Выбор дополнительных тикеров для анализа корреляции")
        multi_tickers = st.text_input("Введите тикеры через запятую (например: AAPL, MSFT, TSLA):", "")

        tickers_list = []
        if multi_tickers:
            tickers_list = [t.strip() for t in multi_tickers.split(",")]
            tickers_list.append(selected_ticker) 
            multi_data = {}
            for t in tickers_list:
                t_data = yf.Ticker(t).history(period="1y", interval="1d")['Close']
                if not t_data.empty:
                    multi_data[t] = t_data

            if len(multi_data) > 1:
                combined_multi_data = pd.DataFrame(multi_data).dropna()
                corr_matrix = combined_multi_data.corr()

               
                st.header("Тепловая карта корреляции активов")
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.matshow(corr_matrix, cmap="coolwarm")
                fig.colorbar(cax)
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.index)))
                ax.set_xticklabels(corr_matrix.columns, rotation=90)
                ax.set_yticklabels(corr_matrix.index)
                for (i, j), val in np.ndenumerate(corr_matrix):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
                st.pyplot(fig)

                avg_corr = corr_matrix.mean().mean()
                st.write(f"**Средняя корреляция между всеми выбранными активами:** {avg_corr:.2f}")

            else:
                st.warning("Необходимо как минимум два тикера для анализа корреляции.")

        if tickers_list:
            market_index = "^GSPC"  
            market_data = yf.Ticker(market_index).history(period="1y", interval="1d")['Close']

            if not market_data.empty:
                beta_results = []
                for ticker in tickers_list:
                    asset_data = multi_data.get(ticker, None)
                    if asset_data is not None:
                        combined = pd.concat([market_data, asset_data], axis=1).dropna()
                        combined.columns = ['Market', 'Asset']
                        market_return = combined['Market'].pct_change()
                        asset_return = combined['Asset'].pct_change()
                        beta = np.cov(asset_return[1:], market_return[1:])[0, 1] / np.var(market_return[1:])
                        beta_results.append((ticker, beta))


                st.header("Бета коэффициенты для выбранных активов")
                beta_columns = st.columns(len(beta_results))
                for i, (ticker, beta) in enumerate(beta_results):
                    beta_columns[i].metric(label=f"Бета {ticker}", value=f"{beta:.2f}")

                st.write(
                    "Значения рассчитаны на основе индекса S&P 500."
                )


    except Exception as e:
        st.error(f"Ошибка: {e}")
