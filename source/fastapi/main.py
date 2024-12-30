from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pandas as pd
import logging
import os
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

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
    parameters: Optional[dict] = None


class FitResponse(BaseModel):
    model_id: str
    status: str


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
    try:
        order = parameters.get("order", (1, 1, 0))
        model = ARIMA(data, order=order).fit()
        return model
    except Exception as e:
        logging.error(f"Failed to train ARIMA model: {str(e)}")
        raise HTTPException(status_code=500, detail="Training failed.")


def load_data_from_yahoo(ticker: str, period: str) -> List[float]:
    try:
        stock_data = yf.Ticker(ticker)
        history = stock_data.history(period=period)
        if history.empty:
            raise ValueError("No data returned from Yahoo Finance.")
        return history["Close"].tolist()
    except Exception as e:
        logging.error(f"Failed to load data for ticker {ticker}: {str(e)}")
        raise HTTPException(status_code=400,
                            detail="Failed to load data from Yahoo Finance.")


@app.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest):
    if not request.data:
        raise HTTPException(status_code=400,
                            detail="Data is required for training.")
    if request.model_type != "ARIMA":
        raise HTTPException(status_code=400, detail="Unsupported model type.")

    model_id = f"model_{len(MODELS) + 1}"
    parameters = request.parameters or {}

    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(train_arima, request.data, parameters)
            model = future.result(timeout=30)

        MODELS[model_id] = {"model": model, "type": request.model_type}
        logging.info(f"{request.model_type} model " +
                     f"{model_id} trained successfully.")
    except Exception as e:
        logging.error(f"Failed to train model: {str(e)}")
        raise HTTPException(status_code=500, detail="Training failed.")

    return FitResponse(model_id=model_id, status="trained")


@app.post("/fit_yahoo", response_model=FitResponse)
async def fit_yahoo(request: FitYahooFinanceRequest):
    try:
        data = load_data_from_yahoo(request.ticker, request.period)
        if not data or len(data) < 10:
            raise HTTPException(status_code=400,
                                detail="Insufficient data for training.")
    except Exception as e:
        logging.error(f"Data loading failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to load data.")

    model_id = f"yahoo_model_{len(MODELS) + 1}"
    parameters = request.parameters or {}

    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(train_arima, data, parameters)
            model = future.result(timeout=30)

        MODELS[model_id] = {"model": model, "type": "ARIMA"}
        logging.info(f"ARIMA model {model_id} trained " +
                     "successfully using Yahoo Finance data.")
    except Exception as e:
        logging.error(f"Failed to train model: {str(e)}")
        raise HTTPException(status_code=500, detail="Training failed.")

    return FitResponse(model_id=model_id, status="trained")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model_info = MODELS.get(request.model_id)
    if not model_info:
        raise HTTPException(status_code=404,
                            detail=f"Model {request.model_id} not found.")

    model_type = model_info["type"]
    model = model_info["model"]

    try:
        if model_type == "ARIMA":
            forecast = model.forecast(steps=request.steps).tolist()
        else:
            raise HTTPException(status_code=400,
                                detail="Unsupported model type.")
        logging.info(f"Prediction made using model {request.model_id}.")
        return PredictResponse(predictions=forecast)
    except Exception as e:
        logging.error(f"Prediction failed for model " +
                      f"{request.model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
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
    global ACTIVE_MODEL_ID
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found.")
    ACTIVE_MODEL_ID = model_id
    logging.info(f"Active model set to {model_id}.")
    return {"status": "active model set", "model_id": model_id}


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode()))
        data_column = df.columns[0]
        data = df[data_column].tolist()
        return {"status": "success", "data": data[:10],
                "message": "First 10 rows of dataset loaded."}
    except Exception as e:
        logging.error(f"Failed to upload dataset: {str(e)}")
        raise HTTPException(status_code=400,
                            detail="Failed to process dataset.")


@app.on_event("startup")
async def load_default_model():
    global ACTIVE_MODEL_ID
    default_data = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    default_parameters = {"order": [1, 1, 0]}
    model = train_arima(default_data, default_parameters)
    model_id = "default_model"
    MODELS[model_id] = {"model": model, "type": "ARIMA"}
    ACTIVE_MODEL_ID = model_id
    logging.info(f"Default ARIMA model {model_id} loaded.")
