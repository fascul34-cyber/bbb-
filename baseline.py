"""
Baseline модель прогнозирования
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


class BaselineModel:
    """Простая baseline модель на основе среднего/медианы"""
    
    def __init__(self, method: str = 'mean'):
        """
        Инициализация
        
        Args:
            method: 'mean' - среднее, 'median' - медиана, 'last' - последнее значение
        """
        self.method = method
        self.fitted_values = {}
    
    def fit(self, data: pd.DataFrame, unified_code: str):
        """
        Обучение модели
        
        Args:
            data: DataFrame с колонками 'date' и 'quantity'
            unified_code: Унифицированный код продукта
        """
        if data.empty or 'quantity' not in data.columns:
            self.fitted_values[unified_code] = 0
            return
        
        if self.method == 'mean':
            value = data['quantity'].mean()
        elif self.method == 'median':
            value = data['quantity'].median()
        elif self.method == 'last':
            value = data['quantity'].iloc[-1] if len(data) > 0 else 0
        else:
            value = data['quantity'].mean()
        
        self.fitted_values[unified_code] = value if not pd.isna(value) else 0
    
    def predict(self, unified_code: str, periods: int = 18) -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            periods: Количество периодов для прогноза
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.fitted_values:
            return np.zeros(periods)
        
        value = self.fitted_values[unified_code]
        return np.full(periods, value)
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        return f"Baseline ({self.method})"


