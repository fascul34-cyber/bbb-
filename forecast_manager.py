"""
Модуль для управления прогнозами и хранения исторических данных
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class ForecastManager:
    """Класс для управления прогнозами"""
    
    def __init__(self, storage_path: str = "forecast_history"):
        """
        Инициализация
        
        Args:
            storage_path: Путь для хранения исторических прогнозов
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_forecast(self, forecast: pd.DataFrame, model_name: str,
                     marketplace: str, forecast_date: datetime = None,
                     metadata: Dict = None):
        """
        Сохраняет прогноз в историю
        
        Args:
            forecast: DataFrame с прогнозом
            model_name: Название модели
            marketplace: 'wb' или 'ozon'
            forecast_date: Дата создания прогноза
            metadata: Дополнительные метаданные
        """
        if forecast_date is None:
            forecast_date = datetime.now()
        
        # Создаем имя файла
        filename = f"{marketplace}_{model_name}_{forecast_date.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.storage_path / filename
        
        # Сохраняем прогноз
        forecast.to_csv(filepath, index=False)
        
        # Сохраняем метаданные
        if metadata:
            metadata_filename = f"{marketplace}_{model_name}_{forecast_date.strftime('%Y%m%d_%H%M%S')}_metadata.json"
            metadata_filepath = self.storage_path / metadata_filename
            
            with open(metadata_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    
    def load_forecast(self, marketplace: str, model_name: str,
                     forecast_date: str = None) -> pd.DataFrame:
        """
        Загружает прогноз из истории
        
        Args:
            marketplace: 'wb' или 'ozon'
            model_name: Название модели
            forecast_date: Дата прогноза (формат: YYYYMMDD_HHMMSS)
        
        Returns:
            DataFrame с прогнозом
        """
        if forecast_date:
            filename = f"{marketplace}_{model_name}_{forecast_date}.csv"
        else:
            # Загружаем последний прогноз
            pattern = f"{marketplace}_{model_name}_*.csv"
            files = list(self.storage_path.glob(pattern))
            if not files:
                return pd.DataFrame()
            filename = max(files, key=lambda p: p.stat().st_mtime).name
        
        filepath = self.storage_path / filename
        if not filepath.exists():
            return pd.DataFrame()
        
        return pd.read_csv(filepath)
    
    def get_forecast_history(self, marketplace: str = None,
                            model_name: str = None,
                            unified_code: str = None) -> pd.DataFrame:
        """
        Получает историю прогнозов
        
        Args:
            marketplace: Фильтр по маркетплейсу
            model_name: Фильтр по модели
            unified_code: Фильтр по продукту
        
        Returns:
            DataFrame с историей прогнозов
        """
        all_forecasts = []
        
        # Ищем все файлы прогнозов
        pattern = "*_*_*.csv"
        if marketplace:
            pattern = f"{marketplace}_*_*.csv"
        if model_name:
            pattern = f"*_{model_name}_*.csv"
        
        files = list(self.storage_path.glob(pattern))
        
        for filepath in files:
            try:
                forecast = pd.read_csv(filepath)
                
                # Извлекаем информацию из имени файла
                parts = filepath.stem.split('_')
                if len(parts) >= 3:
                    forecast['marketplace'] = parts[0]
                    forecast['model_name'] = parts[1]
                    forecast['forecast_date'] = '_'.join(parts[2:])
                
                all_forecasts.append(forecast)
            except Exception as e:
                print(f"Ошибка загрузки {filepath}: {e}")
        
        if not all_forecasts:
            return pd.DataFrame()
        
        result = pd.concat(all_forecasts, ignore_index=True)
        
        # Применяем фильтры
        if unified_code:
            result = result[result['unified_code'] == unified_code]
        
        return result
    
    def compare_forecasts(self, marketplace: str, unified_code: str,
                         actual_sales: pd.DataFrame = None) -> pd.DataFrame:
        """
        Сравнивает прогнозы разных моделей для продукта
        
        Args:
            marketplace: Маркетплейс
            unified_code: Унифицированный код продукта
            actual_sales: Реальные продажи для сравнения
        
        Returns:
            DataFrame с сравнением прогнозов
        """
        # Загружаем все прогнозы для продукта
        forecasts = self.get_forecast_history(
            marketplace=marketplace,
            unified_code=unified_code
        )
        
        if forecasts.empty:
            return pd.DataFrame()
        
        # Группируем по моделям и датам
        comparison = forecasts.groupby(['model_name', 'date']).agg({
            'quantity': 'mean'  # Если несколько прогнозов, берем среднее
        }).reset_index()
        
        # Если есть реальные продажи, добавляем их
        if actual_sales is not None and not actual_sales.empty:
            actual = actual_sales[
                actual_sales['unified_code'] == unified_code
            ].copy()
            if not actual.empty:
                actual = actual.groupby('date')['quantity'].sum().reset_index()
                actual['model_name'] = 'Actual'
                comparison = pd.concat([comparison, actual], ignore_index=True)
        
        return comparison
    
    def get_forecast_metadata(self, marketplace: str, model_name: str,
                             forecast_date: str) -> Dict:
        """
        Загружает метаданные прогноза
        
        Args:
            marketplace: Маркетплейс
            model_name: Название модели
            forecast_date: Дата прогноза
        
        Returns:
            Словарь с метаданными
        """
        metadata_filename = f"{marketplace}_{model_name}_{forecast_date}_metadata.json"
        metadata_filepath = self.storage_path / metadata_filename
        
        if not metadata_filepath.exists():
            return {}
        
        with open(metadata_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


