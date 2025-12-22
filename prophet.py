"""
Prophet модель прогнозирования
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: prophet не установлен. Prophet модель недоступна.")


class ProphetModel:
    """Prophet модель прогнозирования"""
    
    def __init__(self, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 holidays: pd.DataFrame = None):
        """
        Инициализация
        
        Args:
            yearly_seasonality: Включить годовую сезонность
            weekly_seasonality: Включить недельную сезонность
            daily_seasonality: Включить дневную сезонность
            holidays: DataFrame с праздниками (ds, holiday)
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet не установлен")
        
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.models = {}
    
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
        
        if len(data) < 2:
            self.models[unified_code] = None
            return
        
        try:
            # Подготовка данных для Prophet
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(data['date']),
                'y': data['quantity'].fillna(0).values
            })
            
            # Создание модели
            model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=self.holidays
            )
            
            model.fit(prophet_data)
            self.models[unified_code] = model
            
        except Exception as e:
            print(f"Ошибка обучения Prophet для {unified_code}: {e}")
            self.models[unified_code] = None
    
    def predict(self, unified_code: str, periods: int = 18, 
                freq: str = 'D') -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            periods: Количество периодов для прогноза
            freq: Частота ('D' - дни, 'M' - месяцы)
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.models or self.models[unified_code] is None:
            return np.zeros(periods)
        
        try:
            # Создание будущих дат
            if freq == 'D':
                future_periods = periods * 30  # Примерно 30 дней в месяце
            else:
                future_periods = periods
            
            future = self.models[unified_code].make_future_dataframe(
                periods=future_periods, freq=freq
            )
            
            # Прогноз
            forecast = self.models[unified_code].predict(future)
            
            # Берем только будущие значения
            predictions = forecast['yhat'].tail(future_periods).values
            predictions = np.maximum(predictions, 0)  # Отрицательные значения не имеют смысла
            
            # Если нужны месячные значения, агрегируем
            if freq == 'D' and periods < future_periods:
                # Агрегируем по месяцам
                monthly_predictions = []
                for i in range(periods):
                    start_idx = i * 30
                    end_idx = min((i + 1) * 30, len(predictions))
                    monthly_predictions.append(predictions[start_idx:end_idx].sum())
                return np.array(monthly_predictions)
            
            return predictions[:periods]
            
        except Exception as e:
            print(f"Ошибка прогноза Prophet для {unified_code}: {e}")
            return np.zeros(periods)
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        return "Prophet"


