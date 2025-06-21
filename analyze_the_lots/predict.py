import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Подготовка признаков для модели"""
    # Базовая обработка
    try:
        # Преобразование amount в числовой формат
        if df['amount'].dtype == 'object':
            df['amount_parsed'] = df['amount'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
        else:
            df['amount_parsed'] = df['amount'].astype(float)
    except Exception as e:
        print(f"Ошибка при обработке amount: {str(e)}")
        df['amount_parsed'] = df['amount']

    # Преобразование quantity в числовой формат
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    
    # Расчет цены за единицу
    df['unit_price'] = np.where(
        df['quantity'] > 0,
        df['amount_parsed'] / df['quantity'],
        df['amount_parsed']
    )
    
    # Расчет статистик по ценам
    subject_prices = defaultdict(list)
    for _, row in df.iterrows():
        subject = row['subject']
        if row['quantity'] > 0:
            subject_prices[subject].append(row['unit_price'])

    avg_prices = {}
    std_prices = {}
    for subject, prices in subject_prices.items():
        if len(prices) >= 3:
            avg_prices[subject] = np.mean(prices)
            std_prices[subject] = np.std(prices)
        else:
            avg_prices[subject] = None
            std_prices[subject] = None

    df['avg_unit_price'] = df['subject'].map(avg_prices)
    df['std_unit_price'] = df['subject'].map(std_prices)
    
    df['price_deviation'] = np.where(
        (df['avg_unit_price'].notnull()) & (df['std_unit_price'].notnull()),
        (df['unit_price'] - df['avg_unit_price']) / df['std_unit_price'],
        np.nan
    )
    
    df['price_ratio'] = np.where(
        df['avg_unit_price'].notnull(),
        df['unit_price'] / df['avg_unit_price'],
        np.nan
    )
    
    # Статистики по объявлениям
    announcement_groups = df.groupby('announcement')
    announcement_stats = []
    
    for name, group in announcement_groups:
        stats = {
            'announcement': name,
            'lot_count': len(group),
            'max_amount': group['amount_parsed'].max(),
            'min_amount': group['amount_parsed'].min(),
            'total_amount': group['amount_parsed'].sum(),
            'unique_subjects': len(group['subject'].unique()),
            'amount_std': group['amount_parsed'].std(),
            'amount_mean': group['amount_parsed'].mean()
        }
        announcement_stats.append(stats)
    
    announcement_df = pd.DataFrame(announcement_stats)
    announcement_stats = announcement_df.set_index('announcement')
    df = df.join(announcement_stats, on='announcement', rsuffix='_announcement')
    
    # Признаки для ML
    df['amount_round_1000'] = (df['amount_parsed'] % 1000 == 0).astype(int)
    df['amount_round_5000'] = (df['amount_parsed'] % 5000 == 0).astype(int)
    df['amount_round_10000'] = (df['amount_parsed'] % 10000 == 0).astype(int)
    df['amount_round_50000'] = (df['amount_parsed'] % 50000 == 0).astype(int)
    df['amount_round_100000'] = (df['amount_parsed'] % 100000 == 0).astype(int)
    
    # Создание правил для меток
    rules = {}
    rules['high_price'] = np.where(
        (df['avg_unit_price'].notnull()) & 
        (df['unit_price'] > 3 * df['avg_unit_price']),
        1, 0
    )
    
    rules['split'] = np.where(
        (df['lot_count'] >= 3) & 
        (df['max_amount'] < 1000000) &
        (df['total_amount'] >= 3000000),
        1, 0
    )
    
    rules['round'] = np.where(
        (df['amount_parsed'] % 10000 == 0) & 
        (df['amount_parsed'] >= 100000),
        1, 0
    )
    
    rules['suspicious'] = np.where(
        (rules['high_price'] == 1) |
        (rules['split'] == 1) |
        (rules['round'] == 1),
        1, 0
    )
    
    feature_columns = [
        'unit_price', 'quantity', 'amount_parsed',
        'price_deviation', 'price_ratio',
        'lot_count', 'max_amount', 'min_amount', 'total_amount',
        'unique_subjects', 'amount_std', 'amount_mean',
        'amount_round_1000', 'amount_round_5000', 'amount_round_10000',
        'amount_round_50000', 'amount_round_100000'
    ]
    
    # Преобразование всех признаков в числовой формат
    features = df[feature_columns].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features = features.fillna(-999)
    
    return features, rules['suspicious'], rules

def train_models(X_train, y_train):
    """Обучение моделей"""
    models = {
        'xgboost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'catboost': CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            verbose=False
        ),
        'randomforest': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def predict_suspicious(df, models=None, scaler=None):
    """Предсказание подозрительности для новых данных"""
    # Подготовка признаков
    X = prepare_features(df)[0]
    
    # Если модели не переданы, загружаем сохраненные
    if models is None:
        models = {}
        for model_name in ['xgboost', 'lightgbm', 'catboost', 'randomforest']:
            model_path = f'models/{model_name}_model.joblib'
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Модель {model_name} не найдена")
    
    # Если scaler не передан, загружаем сохраненный
    if scaler is None:
        scaler_path = 'models/scaler.joblib'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            raise FileNotFoundError("Scaler не найден")
    
    # Масштабирование признаков
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Получение предсказаний от каждой модели
    predictions = {}
    weights = {
        'xgboost': 0.3,
        'lightgbm': 0.3,
        'catboost': 0.2,
        'randomforest': 0.2
    }
    
    for name, model in models.items():
        predictions[name] = model.predict_proba(X_scaled)[:, 1]
    
    # Взвешенное голосование
    suspicion_probability = sum(predictions[name] * weight 
                              for name, weight in weights.items()) * 100
    
    return suspicion_probability

if __name__ == "__main__":
    # Загрузка данных
    with open('goszakup_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Подготовка признаков и меток
    X, y, rules = prepare_features(df)
    
    # Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Обучение моделей
    models = train_models(X_train, y_train)
    
    # Создание директории для сохранения моделей
    os.makedirs('models', exist_ok=True)
    
    # Сохранение моделей и scaler
    for name, model in models.items():
        joblib.dump(model, f'models/{name}_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Получение предсказаний
    df['suspicion_probability'] = predict_suspicious(df, models, scaler)
    
    # Определение бинарных признаков
    df['is_high_price'] = np.where(df['suspicion_probability'] >= 75, 1, 0)
    df['is_split'] = rules['split']
    df['is_round_amount'] = rules['round']
    df['is_suspicious'] = np.where(df['suspicion_probability'] >= 75, 1, 0)
    
    # Сохранение результатов
    output_columns = [
        'lot_id', 'announcement', 'customer', 'subject', 'quantity',
        'amount', 'purchase_type', 'status', 'subject_link', 'amount_parsed',
        'unit_price', 'avg_unit_price', 'price_deviation', 'suspicion_probability',
        'is_high_price', 'is_split', 'is_round_amount', 'is_suspicious'
    ]
    
    df[output_columns].to_csv('suspicious_purchases.csv', index=False, encoding='utf-8-sig')
    
    print("\nАнализ завершен. Результаты сохранены в suspicious_purchases.csv")
    print(f"\nВсего закупок: {len(df):,}")
    
    print("\nВажность признаков по моделям:")
    print("-" * 50)
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_name.upper()}:")
            importances = dict(zip(X.columns, model.feature_importances_))
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"   {feature}: {importance:.3f}")
    
    print("\nРаспределение вероятностей подозрительности:")
    print("-" * 50)
    ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    for start, end in ranges:
        count = ((df['suspicion_probability'] >= start) & 
                (df['suspicion_probability'] < end)).sum()
        print(f"{start}-{end}%: {count:,} закупок ({count/len(df):.1%})")
    
    print("\nТоп-10 самых подозрительных закупок:")
    print("-" * 50)
    top_suspicious = df.nlargest(10, 'suspicion_probability')[
        ['subject', 'amount_parsed', 'quantity', 'unit_price', 'suspicion_probability']
    ]
    for _, row in top_suspicious.iterrows():
        print(f"\nПредмет: {row['subject'][:100]}...")
        print(f"Сумма: {row['amount_parsed']:,.2f}")
        print(f"Количество: {row['quantity']:.2f}")
        print(f"Цена за единицу: {row['unit_price']:.2f}")
        print(f"Вероятность подозрительности: {row['suspicion_probability']:.1f}%")