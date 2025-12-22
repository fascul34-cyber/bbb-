"""
ARIMA/SARIMA/SARIMAX модели прогнозирования
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels не установлен. ARIMA модели недоступны.")


class ARIMAModel:
    """ARIMA модель прогнозирования"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Tuple[int, int, int, int] = None):
        """
        Инициализация
        
        Args:
            order: (p, d, q) параметры ARIMA
            seasonal_order: (P, D, Q, s) параметры сезонности для SARIMA
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels не установлен")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.is_sarima = seasonal_order is not None
    
    def fit(self, data: pd.DataFrame, unified_code: str):
        """
        Обучение модели
        
        Args:
            data: DataFrame с колонками 'date' и 'quantity'
            unified_code: Унифицированный код продукта
        """
        if data.empty or 'quantity' not in data.columns:
            self.models[unified_code] = None
            return
        
        if len(data) < max(self.order) + 1:
            self.models[unified_code] = None
            return
        
        try:
            ts = data['quantity'].fillna(0).values
            
            if self.is_sarima:
                model = SARIMAX(ts, order=self.order, seasonal_order=self.seasonal_order)
            else:
                model = ARIMA(ts, order=self.order)
            
            fitted_model = model.fit(disp=False)
            self.models[unified_code] = fitted_model
            
        except Exception as e:
            print(f"Ошибка обучения ARIMA для {unified_code}: {e}")
            self.models[unified_code] = None
    
    def predict(self, unified_code: str, periods: int = 18) -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            periods: Количество периодов для прогноза
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.models or self.models[unified_code] is None:
            return np.zeros(periods)
        
        try:
            forecast = self.models[unified_code].forecast(steps=periods)
            forecast = np.maximum(forecast, 0)  # Отрицательные значения не имеют смысла
            return forecast
        except Exception as e:
            print(f"Ошибка прогноза ARIMA для {unified_code}: {e}")
            return np.zeros(periods)
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        if self.is_sarima:
            return f"SARIMA{self.order}x{self.seasonal_order}"
        return f"ARIMA{self.order}"


class SARIMAXModel:
    """SARIMAX модель с экзогенными переменными"""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Инициализация
        
        Args:
            order: (p, d, q) параметры ARIMA
            seasonal_order: (P, D, Q, s) параметры сезонности
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels не установлен")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.exog_columns = []
    
    def fit(self, data: pd.DataFrame, unified_code: str, 
            exog_columns: list = None):
        """
        Обучение модели
        
        Args:
            data: DataFrame с колонками 'date', 'quantity' и экзогенными переменными
            unified_code: Унифицированный код продукта
            exog_columns: Список колонок с экзогенными переменными
        """
        if data.empty or 'quantity' not in data.columns:
            self.models[unified_code] = None
            return
        
        if len(data) < max(self.order) + max(self.seasonal_order[:3]) + 1:
            self.models[unified_code] = None
            return
        
        try:
            ts = data['quantity'].fillna(0).values
            
            if exog_columns:
                exog = data[exog_columns].fillna(0).values
                self.exog_columns = exog_columns
            else:
                exog = None
                self.exog_columns = []
            
            model = SARIMAX(ts, exog=exog, order=self.order, 
                          seasonal_order=self.seasonal_order)
            fitted_model = model.fit(disp=False)
            self.models[unified_code] = fitted_model
            
        except Exception as e:
            print(f"Ошибка обучения SARIMAX для {unified_code}: {e}")
            self.models[unified_code] = None
    
    def predict(self, unified_code: str, future_exog: pd.DataFrame = None, 
                periods: int = 18) -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            future_exog: DataFrame с экзогенными переменными для будущих дат
            periods: Количество периодов для прогноза
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.models or self.models[unified_code] is None:
            return np.zeros(periods)
        
        try:
            if self.exog_columns and future_exog is not None:
                exog = future_exog[self.exog_columns].fillna(0).values[:periods]
            else:
                exog = None
            
            forecast = self.models[unified_code].forecast(steps=periods, exog=exog)
            forecast = np.maximum(forecast, 0)  # Отрицательные значения не имеют смысла
            return forecast
        except Exception as e:
            print(f"Ошибка прогноза SARIMAX для {unified_code}: {e}")
            return np.zeros(periods)
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        return f"SARIMAX{self.order}x{self.seasonal_order}"


