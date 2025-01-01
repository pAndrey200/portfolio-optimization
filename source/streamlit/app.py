import os
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import numpy as np
import requests
import matplotlib.pyplot as plt

if not os.path.exists("logs"):
    os.makedirs("logs")

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=2
)
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.INFO)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

NEWS_API_KEY = "b4efb84bfc22478fb8ca7308c585894a"

API_URL = "http://5.187.3.156:8000"

if "saved_summaries" not in st.session_state:
    st.session_state["saved_summaries"] = {}

if "default_arima_ticker" not in st.session_state:
    st.session_state["default_arima_ticker"] = None
if "default_arima_model_id" not in st.session_state:
    st.session_state["default_arima_model_id"] = None

def get_financial_news(query, api_key, max_results=3):
    financial_domains = (
        "bloomberg.com,cnbc.com,reuters.com,wsj.com,"
        "marketwatch.com,ft.com,yahoo.com,forbes.com,investopedia.com"
    )
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&apiKey={api_key}&language=en&sortBy=publishedAt&domains={financial_domains}"
    )
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        articles = response.json().get("articles", [])[:max_results]
        return articles
    return []

st.title("AI24: Project, team 31")

tab1, tab2 = st.tabs(["Анализ акций", "Прогнозирование цен"])

with tab1:
    popular_tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "ASML", "NVDA", "QCOM", "QUBT"]
    st.header("Выберите тикер актива")

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None

    cols = st.columns(len(popular_tickers))
    for i, ticker in enumerate(popular_tickers):
        if cols[i].button(ticker, key=f"analysis_{ticker}"):
            st.session_state.selected_ticker = ticker

    user_ticker = st.text_input("Или введите свой тикер:", "", key="analysis_input")
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

            market_cap = info.get("marketCap", None)
            if market_cap:
                market_cap_bln = market_cap / 1e9
                st.write(f"**Оценочная стоимость:** {market_cap_bln:.2f} млрд долларов")
            else:
                st.write("**Рыночная капитализация:** Неизвестно")

            price = info.get("regularMarketPrice", None)
            if not price:
                data_tmp = stock.history(period="1d", interval="1d")
                if not data_tmp.empty:
                    price = data_tmp["Close"].iloc[-1]
            if price:
                st.write(f"**Текущая цена:** ${price:.2f}")
            else:
                st.write("**Текущая цена:** Неизвестно")

            st.header("График цены")
            period = st.selectbox("Выберите период:", ["1mo","3mo","6mo","1y","5y","max"],
                                  index=2, key="analysis_period")
            interval = st.selectbox("Выберите интервал:", ["1d","1wk","1mo"],
                                    index=0, key="analysis_interval")
            data = stock.history(period=period, interval=interval)

            if not data.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Цена закрытия")
                )
                fig.update_layout(
                    title=f"График цены для {selected_ticker}",
                    xaxis_title="Дата",
                    yaxis_title="Цена закрытия",
                    xaxis_rangeslider_visible=True,
                    template="plotly_dark",
                )
                st.plotly_chart(fig)

                st.header("Финансовые новости")
                st.write(f"Последние новости о {selected_ticker}:")
                news = get_financial_news(selected_ticker, NEWS_API_KEY)
                if news:
                    for idx, article in enumerate(news):
                        st.write(f"**[{article['title']}]({article['url']})**", key=f"news_{idx}")
                        st.write(
                            f"*{article['source']['name']}* | Опубликовано: {article['publishedAt']}",
                            key=f"news_source_{idx}",
                        )
                        st.write(f"Описание: {article['description']}", key=f"news_description_{idx}")
                else:
                    st.write("Нет доступных новостей.")

                st.header("Ключевые численные показатели")
                c1, c2, c3 = st.columns(3)
                c1.metric("Средняя цена закрытия", f"{data['Close'].mean():.2f}")
                c2.metric("Медианная цена закрытия", f"{data['Close'].median():.2f}")
                c3.metric("Максимальная цена закрытия", f"{data['Close'].max():.2f}")

                data["Daily Return"] = data["Close"].pct_change()
                volatility = np.std(data["Daily Return"]) * np.sqrt(len(data))
                st.write(f"**Годовая волатильность:** {volatility:.2%}")

                st.header("Японские свечи и Bollinger bands")
                candle = go.Figure(
                    data=[
                        go.Candlestick(
                            x=data.index,
                            open=data["Open"],
                            high=data["High"],
                            low=data["Low"],
                            close=data["Close"],
                            name="Candlesticks",
                        )
                    ]
                )
                candle.update_layout(
                    title=f"Японские свечи для {selected_ticker}",
                    xaxis_title="Дата",
                    yaxis_title="Цена",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                )
                st.plotly_chart(candle)

                data["Bollinger_Mid"] = data["Close"].rolling(window=20).mean()
                data["Bollinger_Up"] = data["Bollinger_Mid"] + (data["Close"].rolling(window=20).std() * 2)
                data["Bollinger_Low"] = data["Bollinger_Mid"] - (data["Close"].rolling(window=20).std() * 2)

                fig_bollinger = go.Figure()
                fig_bollinger.add_trace(
                    go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Цена закрытия")
                )
                fig_bollinger.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Bollinger_Up"],
                        mode="lines",
                        name="Bollinger Верхний",
                        line={"dash": "dot"},
                    )
                )
                fig_bollinger.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Bollinger_Low"],
                        mode="lines",
                        name="Bollinger Нижний",
                        line={"dash": "dot"},
                    )
                )
                fig_bollinger.update_layout(
                    title=f"Bollinger bands для {selected_ticker}",
                    xaxis_title="Дата",
                    yaxis_title="Цена закрытия",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_bollinger)

                if len(data) >= 60:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    result = seasonal_decompose(data["Close"], model="additive", period=30)
                    st.header("Сезонность и Остатки")

                    fig_decompose_1 = go.Figure()
                    fig_decompose_1.add_trace(
                        go.Scatter(
                            x=result.observed.index,
                            y=result.observed,
                            mode="lines",
                            name="Оригинальный ряд",
                            line={"color": "royalblue"},
                        )
                    )
                    fig_decompose_1.add_trace(
                        go.Scatter(
                            x=result.trend.index,
                            y=result.trend,
                            mode="lines",
                            name="Тренд",
                            line={"color": "orange"},
                        )
                    )
                    fig_decompose_1.update_layout(
                        title=f"Оригинальный ряд и Тренд для {selected_ticker}",
                        xaxis_title="Дата",
                        yaxis_title="Значение",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_decompose_1)

                    fig_decompose_2 = go.Figure()
                    fig_decompose_2.add_trace(
                        go.Scatter(
                            x=result.seasonal.index,
                            y=result.seasonal,
                            mode="lines",
                            name="Сезонность",
                            line={"color": "green"},
                        )
                    )
                    fig_decompose_2.add_trace(
                        go.Scatter(
                            x=result.resid.index,
                            y=result.resid,
                            mode="lines",
                            name="Остатки",
                            line={"color": "red"},
                        )
                    )
                    fig_decompose_2.update_layout(
                        title=f"Сезонность и Остатки для {selected_ticker}",
                        xaxis_title="Дата",
                        yaxis_title="Значение",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_decompose_2)

        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab2:
    st.header("Прогнозирование цен акций")

    if "selected_ticker_forecast" not in st.session_state:
        st.session_state.selected_ticker_forecast = None

    popular_tickers_2 = ["AAPL", "MSFT", "AMZN", "TSLA", "ASML", "NVDA", "QCOM", "QUBT"]
    cols = st.columns(len(popular_tickers_2))
    for i, ticker in enumerate(popular_tickers_2):
        if cols[i].button(ticker, key=f"forecast_{ticker}"):
            st.session_state.selected_ticker_forecast = ticker

    user_ticker_forecast = st.text_input("Или введите свой тикер:", "", key="forecast_input")
    if user_ticker_forecast:
        st.session_state.selected_ticker_forecast = user_ticker_forecast

    if st.session_state.selected_ticker_forecast:
        selected_ticker = st.session_state.selected_ticker_forecast

        if st.session_state.default_arima_ticker != selected_ticker:
            payload_default = {
                "ticker": selected_ticker,
                "period": "1y",
                "model_type": "ARIMA",
                "auto": False,
                "parameters": {
                    "order": [1, 1, 1]
                }
            }
            try:
                res = requests.post(f"{API_URL}/fit_yahoo", json=payload_default, timeout=60)
                if res.status_code == 200:
                    jd = res.json()
                    st.session_state.default_arima_model_id = jd["model_id"]
                    st.session_state["saved_summaries"][jd["model_id"]] = jd["summary"]
                    st.session_state.default_arima_ticker = selected_ticker
                else:
                    st.error(f"Ошибка при обучении дефолтной ARIMA(1,1,1): {res.text}")
            except Exception as exx:
                st.error(f"Ошибка при обучении дефолтной ARIMA(1,1,1): {str(exx)}")

        stock = yf.Ticker(selected_ticker)
        period = st.selectbox(
            "Выберите период данных:", ["1mo", "3mo", "6mo", "1y", "5y", "max"],
            index=4, key="forecast_period"
        )
        steps = st.number_input(
            "Введите количество шагов прогноза:",
            min_value=1, step=1, value=30, key="forecast_steps"
        )
        data = stock.history(period=period, interval="1d")

        if not data.empty:
            st.header(f"График цены закрытия для {selected_ticker}")
            st.line_chart(data["Close"])

            st.header("ACF и PACF")
            lags = 40
            fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
            acf_values = acf(data["Close"].dropna(), nlags=lags)
            ax_acf.bar(range(len(acf_values)), acf_values)
            ax_acf.set_title("Автокорреляционная функция (ACF)")

            fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
            pacf_values = pacf(data["Close"].dropna(), nlags=lags)
            ax_pacf.bar(range(len(pacf_values)), pacf_values)
            ax_pacf.set_title("Частичная автокорреляционная функция (PACF)")

            st.pyplot(fig_acf)
            st.pyplot(fig_pacf)

            st.header("Результаты теста на стационарность (ADF)")
            adf_result = adfuller(data["Close"].dropna())
            st.write("p-value:", adf_result[1])
            if adf_result[1] < 0.05:
                st.write("Временной ряд стационарен.")
            else:
                st.write("Временной ряд нестационарен.")

            st.header("Выберите модель для прогнозирования")
            model_type = st.selectbox("Тип модели", ["ARIMA", "SARIMA", "VAR"], key="forecast_model_type")

            use_auto = False
            p = d = q = 0
            P = D = Q = m = 0

            if model_type in ["ARIMA", "SARIMA"]:
                st.subheader("Параметры модели")
                use_auto = st.checkbox("Использовать автоподбор", key="forecast_auto_checkbox")

                if not use_auto:
                    p = st.number_input("p", min_value=0, step=1, value=1, key="forecast_p")
                    d = st.number_input("d", min_value=0, step=1, value=1, key="forecast_d")
                    q = st.number_input("q", min_value=0, step=1, value=1, key="forecast_q")
                    if model_type == "SARIMA":
                        P = st.number_input("P", min_value=0, step=1, value=1, key="forecast_P")
                        D = st.number_input("D", min_value=0, step=1, value=1, key="forecast_D")
                        Q = st.number_input("Q", min_value=0, step=1, value=1, key="forecast_Q")
                        m = st.number_input("m (сезонность)", min_value=1, step=1, value=12, key="forecast_m")

            if st.button(f"Запустить модель {model_type}", key="forecast_model_button"):
                try:
                    fit_payload = {
                        "ticker": selected_ticker,
                        "period": period,
                        "model_type": model_type,
                        "auto": use_auto
                    }
                    if not use_auto:
                        if model_type == "VAR":
                            pass
                        elif model_type == "SARIMA":
                            fit_payload["parameters"] = {
                                "order": (p, d, q),
                                "seasonal_order": (P, D, Q, m)
                            }
                        else:
                            fit_payload["parameters"] = {
                                "order": (p, d, q)
                            }


                    fit_response = requests.post(f"{API_URL}/fit_yahoo", json=fit_payload, timeout=120)
                    if fit_response.status_code != 200:
                        st.error(f"Ошибка при обучении модели: {fit_response.text}")
                        st.stop()

                    fit_data = fit_response.json()
                    new_model_id = fit_data["model_id"]
                    new_summary = fit_data["summary"]
                    st.session_state["saved_summaries"][new_model_id] = new_summary

                    pred_payload = {"model_id": new_model_id, "steps": steps}
                    pred_resp = requests.post(f"{API_URL}/predict", json=pred_payload, timeout=120)
                    if pred_resp.status_code != 200:
                        st.error(f"Ошибка при прогнозе: {pred_resp.text}")
                        st.stop()
                    predictions = pred_resp.json()["predictions"]

                    future_index = pd.date_range(start=data.index[-1], periods=steps, freq="D")

                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=data.index, y=data["Close"], mode="lines", name="История"
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=future_index, y=predictions, mode="lines",
                        name=f"Прогноз [{new_model_id}]"
                    ))
                    fig_forecast.update_layout(
                        title=f"Прогноз модели {model_type} для {selected_ticker}",
                        xaxis_title="Дата",
                        yaxis_title="Цена",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_forecast)

                    st.write(f"**model_id:** {new_model_id}")
                    st.text(new_summary)

                except Exception as ex_:
                    st.error(f"Ошибка при обращении к API: {str(ex_)}")

            st.write("---")
            if st.button("Показать все модели"):
                try:
                    models_resp = requests.get(f"{API_URL}/models", timeout=30)
                    if models_resp.status_code != 200:
                        st.error(f"Ошибка при загрузке списка моделей: {models_resp.text}")
                    else:
                        models_list = models_resp.json()
                        if not models_list:
                            st.warning("Нет моделей.")
                        else:
                            fig_all = go.Figure()
                            fig_all.add_trace(go.Scatter(
                                x=data.index, y=data["Close"], mode="lines", name="Исторические данные"
                            ))
                            for m_info in models_list:
                                mid = m_info["model_id"]
                                pr_payload = {
                                    "model_id": mid,
                                    "steps": steps
                                }
                                pr_res = requests.post(f"{API_URL}/predict", json=pr_payload, timeout=60)
                                if pr_res.status_code == 200:
                                    preds = pr_res.json()["predictions"]
                                    future_idx = pd.date_range(start=data.index[-1], periods=steps, freq="D")
                                    fig_all.add_trace(go.Scatter(
                                        x=future_idx, y=preds, mode="lines",
                                        name=f"{mid} ({m_info['model_type']})"
                                    ))
                                else:
                                    st.error(f"Ошибка прогнозирования для {mid}: {pr_res.text}")

                            fig_all.update_layout(
                                title="Все модели: прогнозы",
                                xaxis_title="Дата",
                                yaxis_title="Цена",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_all)

                            st.subheader("Summary всех моделей")
                            for m_info in models_list:
                                mid = m_info["model_id"]
                                summ_text = st.session_state["saved_summaries"].get(mid, "Нет сохранённого описания.")
                                st.write(f"**Model ID:** {mid} | Type: {m_info['model_type']} | Status: {m_info['status']}")
                                st.text(summ_text)
                                st.write("---")
                except Exception as ex_:
                    st.error(f"Ошибка при получении списка моделей: {str(ex_)}")


            if st.button("Удалить все модели"):
                try:
                    del_resp = requests.post(f"{API_URL}/delete_all_models", timeout=30)
                    if del_resp.status_code == 200:
                        st.write("Все модели удалены.")

                        st.session_state["saved_summaries"].clear()
                        st.session_state.default_arima_ticker = None
                        st.session_state.default_arima_model_id = None
                    else:
                        st.error(f"Ошибка при удалении моделей: {del_resp.text}")
                except Exception as ex_:
                    st.error(f"Ошибка при обращении к API: {str(ex_)}")