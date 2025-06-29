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
from dotenv import load_dotenv
from fastapi.routing import APIRoute
from starlette.routing import Route, Mount
from fastapi.responses import RedirectResponse
from parsebot.parse_goszakup import main as parse_goszakup_main

load_dotenv()

app = FastAPI(title="Corruption Analysis API")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API работает"}

@app.post("//analyze")
async def analyze_data_double_slash(file: UploadFile):
    return await analyze_data(file)

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
PORT = int(os.getenv("PORT", "8000"))
HOST = "0.0.0.0"

class PredictionResult(BaseModel):
    id: str
    lot_id: str
    customer: str
    subject: str
    subject_link: str
    amount: float
    quantity: float
    suspicion_percentage: float
    suspicion_level: str

class PaginationRequest(BaseModel):
    page: int = 1
    per_page: int = 10

class AnalysisResponse(BaseModel):
    success: bool
    predictions: Optional[List[PredictionResult]] = None
    error: Optional[str] = None
    execution_time: float = 0.0

class PaginationResponse(BaseModel):
    success: bool
    data: List[dict] = []
    total: int = 0
    page: int = 1
    per_page: int = 10
    total_pages: int = 1
    new_lots_count: int = 0
    error: Optional[str] = None

class GoszakupRequest(BaseModel):
    page: int = 1
    count_record: int = 2000

class GoszakupResponse(BaseModel):
    success: bool
    data: List[dict] = []
    error: Optional[str] = None

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
                lot_id=str(row.get('lot_id', '')),
                customer=str(row.get('customer', '')),
                subject=str(row.get('subject', '')),
                subject_link=str(row.get('subject_link', '')),
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

@app.post("/parse_goszakup", response_model=PaginationResponse)
async def parse_goszakup(request: PaginationRequest):
    try:
        from analyze_the_lots.parse_goszakup import main as parse_main
        
        # Запускаем парсинг новых лотов
        new_lots = parse_main(start_page=request.page, max_pages=1)
        
        # Читаем обновленный файл
        with open(os.path.join(os.path.dirname(__file__), 'analyze_the_lots', 'suspicious_purchases.csv'), 'r', encoding='utf-8') as f:
            df = pd.read_csv(f)
        
        total_records = len(df)
        total_pages = (total_records + request.per_page - 1) // request.per_page
        
        start_idx = (request.page - 1) * request.per_page
        end_idx = min(start_idx + request.per_page, total_records)
        
        page_data = df.iloc[start_idx:end_idx].to_dict('records')
        
        return PaginationResponse(
            success=True,
            data=page_data,
            total=total_records,
            page=request.page,
            per_page=request.per_page,
            total_pages=total_pages,
            new_lots_count=len(new_lots)
        )
    except Exception as e:
        return PaginationResponse(
            success=False,
            error=str(e)
        )

@app.post("/parse", response_model=GoszakupResponse)
async def parse_goszakup_endpoint(request: GoszakupRequest):
    try:
        print(f"\n[API] Получен запрос на парсинг. Страница: {request.page}, Количество записей: {request.count_record}")
        start_time = time.time()
        
        data = parse_goszakup_main(page=request.page, count_record=request.count_record)
        
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        print(f"[API] Парсинг завершен. Получено записей: {len(data)}")
        print(f"[API] Время выполнения: {execution_time} секунд")
        
        return GoszakupResponse(
            success=True,
            data=data
        )
    except Exception as e:
        print(f"[API ERROR] Ошибка при парсинге: {str(e)}")
        return GoszakupResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    print(f"Using Python from: {PYTHON_PATH}")
    print(f"Starting server on port: {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 