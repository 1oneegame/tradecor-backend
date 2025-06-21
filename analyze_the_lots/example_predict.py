import pandas as pd
import numpy as np
import json
import sys
from predict import predict_suspicious
import joblib

def parse_amount(amount):
    if amount is None:
        return 0
    try:
        if isinstance(amount, str):
            # Удаляем все пробелы и заменяем запятую на точку
            amount = amount.replace(' ', '').replace(',', '.')
        return float(amount)
    except (ValueError, TypeError):
        return 0

def parse_quantity(quantity):
    if quantity is None:
        return 0
    try:
        if isinstance(quantity, str):
            # Удаляем все пробелы и заменяем запятую на точку
            quantity = quantity.replace(' ', '').replace(',', '.')
        return float(quantity)
    except (ValueError, TypeError):
        return 0

def main():
    try:
        # Читаем JSON из stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(data)
        
        # Загружаем модели
        models = {}
        for model_name in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
            model_path = f'models/{model_name}_model.joblib'
            models[model_name] = joblib.load(model_path)
        
        # Загружаем scaler
        scaler = joblib.load('models/scaler.joblib')
        
        # Получаем предсказания
        predictions = predict_suspicious(df, models, scaler)
        
        # Формируем результат
        results = []
        for i, row in df.iterrows():
            amount = parse_amount(row.get('amount'))
            quantity = parse_quantity(row.get('quantity'))
            
            results.append({
                'id': row.get('id', str(i)),
                'subject': row.get('subject', ''),
                'amount': amount,
                'quantity': quantity,
                'suspicion_percentage': float(predictions[i]),
                'suspicion_level': 'High' if predictions[i] > 70 else ('Medium' if predictions[i] > 30 else 'Low')
            })
        
        # Выводим результат в stdout как JSON
        print(json.dumps(results, ensure_ascii=False))
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'type': str(type(e).__name__)
        }
        print(json.dumps(error_response, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main() 