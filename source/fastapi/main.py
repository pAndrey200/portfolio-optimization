from typing import List, Optional
import logging
import os
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import itertools
from pydantic import BaseModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()


class FitRequest(BaseModel):
    data: List[float]
    model_type: str
    parameters: Optional[dict] = None


class FitYahooFinanceRequest(BaseModel):
    ticker: str
    period: str
    model_type: Optional[str] = "ARIMA"
    parameters: Optional[dict] = None
    auto: Optional[bool] = False


class FitResponse(BaseModel):
    model_id: str
    status: str
    summary: str


class PredictRequest(BaseModel):
    model_id: str
    steps: int


class PredictResponse(BaseModel):
    predictions: List[float]


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    status: str


MODELS = {}
ACTIVE_MODEL_ID = None


def train_arima(data: List[float], parameters: dict) -> ARIMA:
    '''
    Обучение модели arima
    '''
    order = parameters.get("order", (1, 1, 0))
    model = ARIMA(data, order=order).fit()
    return model


def train_sarima(data: List[float], parameters: dict) -> SARIMAX:
    '''
    Обучение модели sarima
    '''
    order = parameters.get("order", (1, 1, 1))
    seasonal_order = parameters.get("seasonal_order", (1, 1, 1, 12))
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order).fit()
    return model


def auto_arima_search(data: List[float], seasonal: bool = False) -> SARIMAX:
    '''
    Подбор гиперпараметров для модели arima
    '''
    best_aic = np.inf
    best_model = None
    if seasonal:
        m = 12
        for p, d, q, P, D, Q in itertools.product(range(0, 3), range(0, 2), range(0, 3),
                                                  range(0, 2), range(0, 2), range(0, 2)):
            try:
                model = SARIMAX(
                    data,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m)
                ).fit(disp=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
            except ValueError:
                continue
    else:
        for p, d, q in itertools.product(range(0, 3), range(0, 2), range(0, 3)):
            try:
                model = SARIMAX(data, order=(p, d, q)).fit(disp=False)
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
            except ValueError:
                continue
    if not best_model:
        raise HTTPException(status_code=500, detail="Auto ARIMA/SARIMA failed.")
    return best_model


def train_var(df: pd.DataFrame) -> VAR:
    '''
    вспомогательная функция для обучения модели
    '''
    try:
        logging.info("train_var: df.shape before any transform = %s", df.shape)
        df = df[['Close', 'Volume', 'High-Low']].copy().dropna()
        logging.info("train_var: df.shape after dropna = %s", df.shape)
        var_model = VAR(df)
        var_fit = var_model.fit(maxlags=2, ic='aic')
        return var_fit
    except Exception as e:
        logging.error("VAR training exception: %s", str(e))
        raise


def load_data_from_yahoo_for_close(ticker: str, period: str) -> List[float]:
    '''
    Загрузка цен активов на момент закрытия
    '''
    stock_data = yf.Ticker(ticker)
    history = stock_data.history(period=period)
    if history.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    return history["Close"].tolist()


def load_data_from_yahoo_for_var(ticker: str, period: str) -> pd.DataFrame:
    '''
    Загрузка цен активов
    '''
    stock_data = yf.Ticker(ticker)
    history = stock_data.history(period=period)
    if history.empty:
        raise ValueError("No data returned from Yahoo Finance.")
    df = history[["Close", "Volume", "High", "Low"]].copy()
    df["High-Low"] = df["High"] - df["Low"]
    return df


@app.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest):
    '''
    обучение модели по данным с датасета
    '''
    if not request.data:
        raise HTTPException(status_code=400, detail="Data is required for training.")
    if request.model_type != "ARIMA":
        raise HTTPException(status_code=400, detail="Unsupported model type.")
    model_id = f"model_{len(MODELS) + 1}"
    parameters = request.parameters or {}
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(train_arima, request.data, parameters)
            model = future.result(timeout=30)
        summary_text = str(model.summary())
        MODELS[model_id] = {"model": model, "type": request.model_type}
    except Exception as e:
        logging.error("ARIMA training exception: %s", str(e))
        raise HTTPException(status_code=500, detail="Training failed.") from e

    return FitResponse(model_id=model_id, status="trained", summary=summary_text)


@app.post("/fit_yahoo", response_model=FitResponse)
async def fit_yahoo(request: FitYahooFinanceRequest):
    '''
    Обучение модели по данным с yahoo.com
    '''
    model_type = request.model_type.upper()
    model_id = f"yahoo_model_{len(MODELS) + 1}"

    if model_type == "VAR":
        df_var = load_data_from_yahoo_for_var(request.ticker, request.period)
        if len(df_var) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for VAR.")
        try:
            with ThreadPoolExecutor() as executor:
                var_fit = executor.submit(train_var, df_var).result(timeout=60)
            MODELS[model_id] = {"model": var_fit, "type": "VAR"}
            return FitResponse(model_id=model_id, status="trained", summary=str(var_fit.summary()))
        except Exception as e:
            logging.error("Failed to train VAR model: %s", str(e))
            raise HTTPException(status_code=500, detail="Training VAR failed.") from e

    else:
        data = load_data_from_yahoo_for_close(request.ticker, request.period)
        if not data or len(data) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for ARIMA/SARIMA.")

        if request.auto:
            seasonal = model_type == "SARIMA"
            try:
                with ThreadPoolExecutor() as executor:
                    best_model = executor.submit(auto_arima_search, data, seasonal).result(timeout=60)
                final_type = "SARIMA" if seasonal else "ARIMA"
                MODELS[model_id] = {"model": best_model, "type": final_type}
                return FitResponse(model_id=model_id, status="trained", summary=str(best_model.summary()))
            except Exception as e:
                logging.error("Auto search exception: %s", str(e))
                raise HTTPException(status_code=500, detail="Auto search failed.") from e
        else:
            parameters = request.parameters or {}
            if model_type == "SARIMA":
                try:
                    with ThreadPoolExecutor() as executor:
                        sarima_model = executor.submit(train_sarima, data, parameters).result(timeout=60)
                    MODELS[model_id] = {"model": sarima_model, "type": "SARIMA"}
                    return FitResponse(model_id=model_id, status="trained", summary=str(sarima_model.summary()))
                except Exception as e:
                    logging.error("SARIMA training exception: %s", str(e))
                    raise HTTPException(status_code=500, detail="Training SARIMA failed.") from e
            else:
                try:
                    with ThreadPoolExecutor() as executor:
                        arima_model = executor.submit(train_arima, data, parameters).result(timeout=60)
                    MODELS[model_id] = {"model": arima_model, "type": "ARIMA"}
                    return FitResponse(model_id=model_id, status="trained", summary=str(arima_model.summary()))
                except Exception as e:
                    logging.error("ARIMA training exception: %s", str(e))
                    raise HTTPException(status_code=500, detail="Training ARIMA failed.") from e


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    '''
    Предсказвает цену актива
    '''
    model_info = MODELS.get(request.model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found.")
    model_type = model_info["type"]
    model = model_info["model"]
    try:
        if model_type in ["ARIMA", "SARIMA"]:
            forecast = model.forecast(steps=request.steps).tolist()
        elif model_type == "VAR":
            last_values = model.endog[-model.k_ar:]
            forecast_diff = model.forecast(last_values, steps=request.steps)
            forecast = forecast_diff[:, 0].tolist()
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type.")
        return PredictResponse(predictions=forecast)
    except Exception as e:
        logging.error("Prediction exception: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed.") from e


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    '''
    возвращает список моделей
    '''
    return [
        ModelInfo(
            model_id=model_id,
            model_type=model_info["type"],
            status="active" if model_id == ACTIVE_MODEL_ID else "ready"
        )
        for model_id, model_info in MODELS.items()
    ]


@app.post("/set_model")
async def set_model(model_id: str):
    '''
    выбор модели для обучения
    '''
    global ACTIVE_MODEL_ID
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found.")
    ACTIVE_MODEL_ID = model_id
    return {"status": "active model set", "model_id": model_id}


@app.post("/delete_all_models")
async def delete_all_models():
    '''
    удаление всех моделей
    '''
    global ACTIVE_MODEL_ID
    MODELS.clear()
    ACTIVE_MODEL_ID = None
    logging.info("All models have been deleted.")
    return {"status": "all models removed"}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    '''
    загрузка данных из файла
    '''
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        data_column = df.columns[0]
        data = df[data_column].tolist()
        return {"status": "success", "data": data[:10], "message": "First 10 rows of dataset loaded."}
    except Exception as e:
        logging.error("Upload dataset exception: %s", str(e))
        raise HTTPException(status_code=400, detail="Failed to process dataset.") from e
