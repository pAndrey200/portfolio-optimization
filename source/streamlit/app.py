import os
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
import numpy as np
import requests
import matplotlib.pyplot as plt

if not os.path.exists("logs"):
    os.makedirs("logs")

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=2)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

NEWS_API_KEY = "b4efb84bfc22478fb8ca7308c585894a"


def get_financial_news(query, api_key, max_results=3):
    financial_domains = "bloomberg.com,cnbc.com,reuters.com,wsj.com,marketwatch.com," +\
                        "ft.com,yahoo.com,forbes.com,investopedia.com"
    url = (
        f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&" +
        f"language=en&sortBy=publishedAt&domains={financial_domains}")
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        articles = response.json().get("articles", [])[:max_results]
        return articles
    return []


st.title("AI24: Project, team 31")

tab1, tab2 = st.tabs(["Анализ акций", "Прогнозирование цен"])

with tab1:
    popular_tickers = ["AAPL", "MSFT", "AMZN",
                       "TSLA", "ASML", "NVDA", "QCOM", "QUBT"]
    st.header("Выберите тикер актива")

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None

    cols = st.columns(len(popular_tickers))
    for i, ticker in enumerate(popular_tickers):
        if cols[i].button(ticker, key=f"analysis_{ticker}"):
            st.session_state.selected_ticker = ticker

    user_ticker = st.text_input(
        "Или введите свой тикер:", "", key="analysis_input")
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
                st.write(
                    f"**Оценочная стоимость:** {market_cap_bln:.2f} " +
                    "млрд долларов")
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
            period = st.selectbox(
                "Выберите период:", [
                    "1mo", "3mo", "6mo", "1y", "5y", "max"],
                index=2,
                key="analysis_period")
            interval = st.selectbox(
                "Выберите интервал:", [
                    "1d", "1wk", "1mo"], index=0, key="analysis_interval")
            data = stock.history(period=period, interval=interval)

            if not data.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Цена закрытия'))
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
                    for idx, article in enumerate(news):
                        st.write(
                            f"**[{article['title']}]({article['url']})**",
                            key=f"news_{idx}")
                        st.write(
                            f"*{article['source']['name']}* | Опубликовано: " +
                            f"{article['publishedAt']}",
                            key=f"news_source_{idx}")
                        st.write(
                            f"Описание: {article['description']}",
                            key=f"news_description_{idx}")
                else:
                    st.write("Нет доступных новостей.")

                st.header("Ключевые численные показатели")
                metrics_columns = st.columns(3)
                metrics_columns[0].metric(
                    label="Средняя цена закрытия",
                    value=f"{data['Close'].mean():.2f}")
                metrics_columns[1].metric(
                    label="Медианная цена закрытия",
                    value=f"{data['Close'].median():.2f}")
                metrics_columns[2].metric(
                    label="Максимальная цена закрытия",
                    value=f"{data['Close'].max():.2f}")

                data['Daily Return'] = data['Close'].pct_change()
                volatility = np.std(data['Daily Return']) * np.sqrt(len(data))
                st.write(f"**Годовая волатильность:** {volatility:.2%}")

                st.header("Японские свечи и Bollinger bands")

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
                fig_bollinger.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Цена закрытия'))
                fig_bollinger.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Bollinger_Up'],
                        mode='lines',
                        name='Bollinger Верхний',
                        line={"dash": 'dot'}))
                fig_bollinger.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Bollinger_Low'],
                        mode='lines',
                        name='Bollinger Нижний',
                        line={"dash": 'dot'}))
                fig_bollinger.update_layout(
                    title=f'Bollinger bands для {selected_ticker}',
                    xaxis_title="Дата",
                    yaxis_title="Цена закрытия",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_bollinger)

                if len(data) >= 60:
                    result = seasonal_decompose(
                        data['Close'], model='additive', period=30)

                    st.header("Сезонность и Остатки")
                    fig_decompose_1 = go.Figure()
                    fig_decompose_1.add_trace(
                        go.Scatter(
                            x=result.observed.index,
                            y=result.observed,
                            mode='lines',
                            name='Оригинальный ряд',
                            line={"color": 'royalblue'}))
                    fig_decompose_1.add_trace(
                        go.Scatter(
                            x=result.trend.index,
                            y=result.trend,
                            mode='lines',
                            name='Тренд',
                            line={"color": 'orange'}))
                    fig_decompose_1.update_layout(
                        title='Оригинальный ряд и Тренд для ' +
                        f'{selected_ticker}',
                        xaxis_title="Дата",
                        yaxis_title="Значение",
                        template="plotly_dark")
                    st.plotly_chart(fig_decompose_1)

                    fig_decompose_2 = go.Figure()
                    fig_decompose_2.add_trace(
                        go.Scatter(
                            x=result.seasonal.index,
                            y=result.seasonal,
                            mode='lines',
                            name='Сезонность',
                            line={"color": 'green'}))
                    fig_decompose_2.add_trace(
                        go.Scatter(
                            x=result.resid.index,
                            y=result.resid,
                            mode='lines',
                            name='Остатки',
                            line={"color": 'red'}))
                    fig_decompose_2.update_layout(
                        title=f'Сезонность и Остатки для {selected_ticker}',
                        xaxis_title="Дата",
                        yaxis_title="Значение",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_decompose_2)

                st.header("Выбор дополнительных тикеров для анализа корреляции")
                multi_tickers = st.text_input(
                    "Введите тикеры через запятую (например: AAPL, MSFT, TSLA):",
                    "",
                    key="correlation_input")

                if multi_tickers:
                    tickers_list = [t.strip()
                                    for t in multi_tickers.split(",")]
                    tickers_list.append(selected_ticker)

                    multi_data = {}
                    for t in tickers_list:
                        t_data = yf.Ticker(t).history(
                            period="1y", interval="1d")['Close']
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
                            ax.text(j, i, f'{val:.2f}', ha='center',
                                    va='center', color='black')
                        st.pyplot(fig)

                        avg_corr = corr_matrix.mean().mean()
                        st.write(
                            "**Средняя корреляция между всеми выбранными " +
                            f"активами:** {avg_corr:.2f}")

                        market_index = "^GSPC"
                        market_data = yf.Ticker(market_index).history(
                            period="1y", interval="1d")['Close']

                        if not market_data.empty:
                            beta_results = []
                            for ticker in tickers_list:
                                asset_data = multi_data.get(ticker, None)
                                if asset_data is not None:
                                    combined = pd.concat(
                                        [market_data, asset_data],
                                        axis=1).dropna()
                                    combined.columns = ['Market', 'Asset']
                                    market_return = combined['Market'].pct_change()
                                    asset_return = combined['Asset'].pct_change(
                                    )
                                    beta = np.cov(asset_return[1:],
                                                  market_return[1:])[0, 1] / np.var(market_return[1:])
                                    beta_results.append((ticker, beta))

                            st.header(
                                "Бета коэффициенты для выбранных активов")
                            beta_columns = st.columns(len(beta_results))
                            for i, (ticker, beta) in enumerate(beta_results):
                                beta_columns[i].metric(
                                    label=f"Бета {ticker}",
                                    value=f"{beta:.2f}")

                    else:
                        st.warning("Необходимо как минимум два тикера для анализа корреляции.")

        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab2:
    st.header("Прогнозирование цен акций")
    if "selected_ticker_forecast" not in st.session_state:
        st.session_state.selected_ticker_forecast = None

    cols = st.columns(len(popular_tickers))
    for i, ticker in enumerate(popular_tickers):
        if cols[i].button(ticker, key=f"forecast_{ticker}"):
            st.session_state.selected_ticker_forecast = ticker

    user_ticker_forecast = st.text_input(
        "Или введите свой тикер:", "", key="forecast_input")
    if user_ticker_forecast:
        st.session_state.selected_ticker_forecast = user_ticker_forecast

    if st.session_state.selected_ticker_forecast:
        selected_ticker = st.session_state.selected_ticker_forecast
        logger.info("Пользователь выбрал тикер: %s", selected_ticker)
        try:
            stock = yf.Ticker(selected_ticker)
            period = st.selectbox(
                "Выберите период данных:",
                ["1mo", "3mo", "6mo", "1y", "5y", "max"],
                index=4,
                key="forecast_period")
            steps = st.number_input(
                "Введите количество шагов прогноза:",
                min_value=1,
                step=1,
                value=30,
                key="forecast_steps")
            data = stock.history(period=period, interval="1d")

            if not data.empty:
                st.header(f"График цены закрытия для {selected_ticker}")
                st.line_chart(data['Close'])

                st.header("ACF и PACF")
                lags = 40

                fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
                acf_values = acf(data['Close'].dropna(), nlags=lags)
                ax_acf.bar(range(len(acf_values)), acf_values)
                ax_acf.set_title("Автокорреляционная функция (ACF)")

                fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
                pacf_values = pacf(data['Close'].dropna(), nlags=lags)
                ax_pacf.bar(range(len(pacf_values)), pacf_values)
                ax_pacf.set_title(
                    "Частичная автокорреляционная функция (PACF)")

                st.pyplot(fig_acf)
                st.pyplot(fig_pacf)

                st.header("Результаты теста на стационарность (ADF)")
                adf_result = adfuller(data['Close'].dropna())
                st.write("p-value:", adf_result[1], key="adf_p_value")
                if adf_result[1] < 0.05:
                    st.write("Временной ряд стационарен.",
                             key="adf_stationary")
                else:
                    st.write("Временной ряд нестационарен.",
                             key="adf_nonstationary")

                st.header("Выберите модель для прогнозирования")
                model_type = st.selectbox(
                    "Тип модели", [
                        "ARIMA", "SARIMA", "VAR"], key="forecast_model_type")

                if model_type == "VAR":
                    st.write(
                        "Будут использоваться переменные: High, Low, Close, Volume")
                    if st.button("Запустить модель VAR",
                                 key="forecast_var_button"):
                        logger.info("Запуск модели VAR")
                        data['High-Low'] = data['High'] - data['Low']
                        data = data.dropna()

                        model_data = data[['Close', 'Volume',
                                           'High-Low']].diff().dropna()
                        var_model = VAR(model_data)
                        var_fit = var_model.fit(maxlags=5, ic='aic')

                        forecast_diff = var_fit.forecast(
                            y=model_data.values[-var_fit.k_ar:], steps=steps)
                        forecast_diff_df = pd.DataFrame(
                            forecast_diff, columns=model_data.columns)

                        forecast = forecast_diff_df.cumsum()
                        forecast['Close'] += data['Close'].iloc[-1]
                        forecast.index = pd.date_range(
                            start=data.index[-1], periods=steps, freq='D')

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='Исторические данные'))
                        fig.add_trace(
                            go.Scatter(
                                x=forecast.index,
                                y=forecast['Close'],
                                mode='lines',
                                name='Прогноз Close'))
                        fig.update_layout(
                            title=f'Прогноз модели VAR для {selected_ticker}',
                            xaxis_title="Дата",
                            yaxis_title="Цена",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig)
                        st.text(var_fit.summary())

                elif model_type in ["ARIMA", "SARIMA"]:
                    st.subheader("Параметры модели")
                    use_auto = st.checkbox(
                        "Использовать автоматический подбор параметров",
                        key="forecast_auto_checkbox")

                    if not use_auto:
                        p = st.number_input(
                            "Параметр p", min_value=0, step=1,
                            value=1, key="forecast_p")
                        d = st.number_input(
                            "Параметр d", min_value=0, step=1,
                            value=1, key="forecast_d")
                        q = st.number_input(
                            "Параметр q", min_value=0, step=1,
                            value=1, key="forecast_q")
                        if model_type == "SARIMA":
                            P = st.number_input(
                                "Параметр P (сезонный)", min_value=0,
                                step=1, value=1, key="forecast_P")
                            D = st.number_input(
                                "Параметр D (сезонный)", min_value=0,
                                step=1, value=1, key="forecast_D")
                            Q = st.number_input(
                                "Параметр Q (сезонный)", min_value=0,
                                step=1, value=1, key="forecast_Q")
                            m = st.number_input(
                                "Период сезонности", min_value=1,
                                step=1, value=12, key="forecast_m")

                    if st.button(
                            f"Запустить модель {model_type}",
                            key="forecast_model_button"):
                        logger.info("Запуск модели %s", model_type)
                        if use_auto:
                            best_aic = np.inf
                            best_order = None
                            best_model = None

                            for p_try in range(0, 4):
                                for d_try in range(0, 2):
                                    for q_try in range(0, 4):
                                        try:
                                            temp_model = SARIMAX(
                                                data['Close'], order=(
                                                    p_try, d_try, q_try)).fit(
                                                disp=False)
                                            if temp_model.aic < best_aic:
                                                best_aic = temp_model.aic
                                                best_order = (
                                                    p_try, d_try, q_try)
                                                best_model = temp_model
                                        except BaseException:
                                            continue

                            st.write(
                                f"Лучшие параметры: {best_order}, " +
                                f"AIC: {best_aic}")
                            forecast = best_model.forecast(steps=steps)
                        else:
                            if model_type == "ARIMA":
                                model = SARIMAX(
                                    data['Close'], order=(p, d, q)).fit()
                            else:
                                model = SARIMAX(
                                    data['Close'], order=(
                                        p, d, q), seasonal_order=(
                                        P, D, Q, m)).fit()

                            forecast = model.forecast(steps=steps)

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='Исторические данные'))
                        fig.add_trace(go.Scatter(x=pd.date_range(
                            start=data.index[-1], periods=steps, freq='D'),
                            y=forecast, mode='lines', name='Прогноз'))
                        fig.update_layout(
                            title=f'Прогноз модели {model_type} ' +
                                  f'для {selected_ticker}',
                            xaxis_title="Дата",
                            yaxis_title="Цена",
                            template="plotly_dark")
                        st.plotly_chart(fig)

                        if not use_auto:
                            st.text(model.summary())
                        else:
                            st.text(best_model.summary())

        except Exception as e:
            logger.error("Ошибка: %s", str(e))
            st.error(f"Ошибка: {e}")
