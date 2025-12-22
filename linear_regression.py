"""
Линейная регрессия с подбором фичей и бинарными признаками
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionModel:
    """Линейная регрессия с подбором фичей"""
    
    def __init__(self, use_feature_selection: bool = True, k_features: int = 10):
        """
        Инициализация
        
        Args:
            use_feature_selection: Использовать ли подбор фичей
            k_features: Количество фичей для выбора
        """
        self.use_feature_selection = use_feature_selection
        self.k_features = k_features
        self.models = {}
        self.scalers = {}
        self.selected_features = {}
        self.feature_names = []
    
    def fit(self, data: pd.DataFrame, unified_code: str, 
            feature_columns: List[str] = None):
        """
        Обучение модели
        
        Args:
            data: DataFrame с признаками и целевой переменной 'quantity'
            unified_code: Унифицированный код продукта
            feature_columns: Список колонок-признаков
        """
        if data.empty or 'quantity' not in data.columns:
            self.models[unified_code] = None
            return
        
        if feature_columns is None:
            # Автоматический выбор признаков
            feature_columns = [col for col in data.columns 
                            if col not in ['date', 'quantity', 'unified_code', 'sku', 'solo_code']]
        
        if not feature_columns:
            self.models[unified_code] = None
            return
        
        X = data[feature_columns].fillna(0)
        y = data['quantity'].fillna(0)
        
        if len(X) < 2:
            self.models[unified_code] = None
            return
        
        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[unified_code] = scaler
        
        # Подбор фичей
        if self.use_feature_selection and len(feature_columns) > self.k_features:
            selector = SelectKBest(f_regression, k=min(self.k_features, len(feature_columns)))
            X_selected = selector.fit_transform(X_scaled, y)
            self.selected_features[unified_code] = selector.get_support()
            selected_feature_names = [feature_columns[i] for i in range(len(feature_columns)) 
                                    if self.selected_features[unified_code][i]]
        else:
            X_selected = X_scaled
            self.selected_features[unified_code] = np.ones(len(feature_columns), dtype=bool)
            selected_feature_names = feature_columns
        
        self.feature_names = selected_feature_names
        
        # Обучение модели
        model = LinearRegression()
        model.fit(X_selected, y)
        self.models[unified_code] = model
    
    def predict(self, unified_code: str, future_features: pd.DataFrame, 
                periods: int = 18) -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            future_features: DataFrame с признаками для будущих дат
            periods: Количество периодов для прогноза
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.models or self.models[unified_code] is None:
            return np.zeros(periods)
        
        if unified_code not in self.scalers:
            return np.zeros(periods)
        
        # Получаем признаки
        feature_columns = [col for col in future_features.columns 
                         if col not in ['date', 'unified_code', 'sku', 'solo_code']]
        
        if not feature_columns:
            return np.zeros(periods)
        
        X = future_features[feature_columns].fillna(0)
        
        # Масштабирование
        X_scaled = self.scalers[unified_code].transform(X)
        
        # Выбор фичей
        if unified_code in self.selected_features:
            X_selected = X_scaled[:, self.selected_features[unified_code]]
        else:
            X_selected = X_scaled
        
        # Прогноз
        predictions = self.models[unified_code].predict(X_selected)
        predictions = np.maximum(predictions, 0)  # Отрицательные значения не имеют смысла
        
        return predictions[:periods]
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        return "Linear Regression (Feature Selection)" if self.use_feature_selection else "Linear Regression"


class BinaryLinearRegressionModel:
    """Линейная регрессия с бинарными признаками"""
    
    def __init__(self):
        """Инициализация"""
        self.models = {}
        self.scalers = {}
        self.feature_names = []
    
    def fit(self, data: pd.DataFrame, unified_code: str):
        """
        Обучение модели
        
        Args:
            data: DataFrame с признаками и целевой переменной 'quantity'
            unified_code: Унифицированный код продукта
        """
        if data.empty or 'quantity' not in data.columns:
            self.models[unified_code] = None
            return
        
        # Выбираем только бинарные признаки
        binary_features = [col for col in data.columns 
                          if col.startswith('is_') or 
                          col.startswith('month_') or 
                          col.startswith('day_of_week_')]
        
        if not binary_features:
            self.models[unified_code] = None
            return
        
        X = data[binary_features].fillna(0)
        y = data['quantity'].fillna(0)
        
        if len(X) < 2:
            self.models[unified_code] = None
            return
        
        # Масштабирование
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[unified_code] = scaler
        self.feature_names = binary_features
        
        # Обучение модели
        model = LinearRegression()
        model.fit(X_scaled, y)
        self.models[unified_code] = model
    
    def predict(self, unified_code: str, future_features: pd.DataFrame, 
                periods: int = 18) -> np.ndarray:
        """
        Прогноз на periods месяцев
        
        Args:
            unified_code: Унифицированный код продукта
            future_features: DataFrame с признаками для будущих дат
            periods: Количество периодов для прогноза
        
        Returns:
            Массив прогнозных значений
        """
        if unified_code not in self.models or self.models[unified_code] is None:
            return np.zeros(periods)
        
        if unified_code not in self.scalers:
            return np.zeros(periods)
        
        # Получаем бинарные признаки
        binary_features = [col for col in future_features.columns 
                          if col.startswith('is_') or 
                          col.startswith('month_') or 
                          col.startswith('day_of_week_')]
        
        if not binary_features:
            return np.zeros(periods)
        
        # Используем только те признаки, которые были в обучении
        available_features = [f for f in self.feature_names if f in binary_features]
        
        if not available_features:
            return np.zeros(periods)
        
        X = future_features[available_features].fillna(0)
        
        # Добавляем недостающие признаки нулями
        missing_features = [f for f in self.feature_names if f not in available_features]
        for feature in missing_features:
            X[feature] = 0
        
        X = X[self.feature_names]
        
        # Масштабирование
        X_scaled = self.scalers[unified_code].transform(X)
        
        # Прогноз
        predictions = self.models[unified_code].predict(X_scaled)
        predictions = np.maximum(predictions, 0)  # Отрицательные значения не имеют смысла
        
        return predictions[:periods]
    
    def get_model_name(self) -> str:
        """Возвращает название модели"""
        return "Linear Regression (Binary Features)"


