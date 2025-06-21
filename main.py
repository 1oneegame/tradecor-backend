from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
from analyze_the_lots.predict import predict_suspicious
import joblib
from typing import List, Optional
from pydantic import BaseModel
import os
import sys

app = FastAPI(title="Corruption Analysis API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Получаем путь к Python
PYTHON_PATH = sys.executable
PORT = int(os.environ.get("PORT", 8000))
HOST = "0.0.0.0"

class PredictionResult(BaseModel):
    id: str
    subject: str
    amount: float
    quantity: float
    suspicion_percentage: float
    suspicion_level: str

class AnalysisResponse(BaseModel):
    success: bool
    predictions: Optional[List[PredictionResult]] = None
    error: Optional[str] = None
    execution_time: float = 0.0  # Добавляем значение по умолчанию и явно указываем тип float

def parse_amount(amount) -> float:
    if amount is None:
        return 0.0
    try:
        if isinstance(amount, str):
            amount = amount.replace(' ', '').replace(',', '.')
        return float(amount)
    except (ValueError, TypeError):
        return 0.0

def parse_quantity(quantity) -> float:
    if quantity is None:
        return 0.0
    try:
        if isinstance(quantity, str):
            quantity = quantity.replace(' ', '').replace(',', '.')
        return float(quantity)
    except (ValueError, TypeError):
        return 0.0

import time

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(file: UploadFile):
    try:
        start_time = time.time()
        print(f"[DEBUG] Starting analysis at {start_time}")
        # Проверка типа файла
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Только JSON файлы разрешены")

        # Чтение файла
        content = await file.read()
        data = json.loads(content.decode())
        
        # Преобразование в DataFrame
        df = pd.DataFrame(data)
        
        # Загрузка моделей
        models = {}
        models_dir = os.path.join(os.path.dirname(__file__), 'analyze_the_lots', 'models')
        for model_name in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
            model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Модель {model_name} не найдена")
            models[model_name] = joblib.load(model_path)
        
        # Загрузка scaler
        scaler_path = os.path.join(models_dir, 'scaler.joblib')
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=500, detail="Scaler не найден")
        scaler = joblib.load(scaler_path)
        
        # Получение предсказаний
        predictions = predict_suspicious(df, models, scaler)
        
        # Формирование результата
        results = []
        for i, row in df.iterrows():
            amount = parse_amount(row.get('amount'))
            quantity = parse_quantity(row.get('quantity'))
            
            results.append(PredictionResult(
                id=str(row.get('id', i)),
                subject=str(row.get('subject', '')),
                amount=amount,
                quantity=quantity,
                suspicion_percentage=float(predictions[i]),
                suspicion_level='High' if predictions[i] > 70 else ('Medium' if predictions[i] > 30 else 'Low')
            ))
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[DEBUG] Analysis completed at {end_time}")
        print(f"[DEBUG] Total execution time: {execution_time} seconds")
        
        return AnalysisResponse(
            success=True,
            predictions=results,
            execution_time=float(round(execution_time, 2))  # Явно преобразуем в float
        )

    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[DEBUG] Error occurred at {end_time}")
        print(f"[DEBUG] Execution time until error: {execution_time} seconds")
        print(f"[DEBUG] Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_time": float(round(execution_time, 2))
            }
        )

if __name__ == "__main__":
    import uvicorn
    print(f"Using Python from: {PYTHON_PATH}")
    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False) 