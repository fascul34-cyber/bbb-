"""
Модуль для оценки качества моделей прогнозирования
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Класс для оценки качества моделей прогнозирования"""
    
    def __init__(self):
        """Инициализация"""
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str, unified_code: str) -> Dict:
        """
        Оценивает качество модели
        
        Args:
            y_true: Реальные значения
            y_pred: Прогнозные значения
            model_name: Название модели
            unified_code: Унифицированный код продукта
        
        Returns:
            Словарь с метриками
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'r2': -np.inf
            }
        
        # Обрезаем до минимальной длины
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Убираем нули и бесконечности
        mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
        if mask.sum() == 0:
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'r2': -np.inf
            }
        
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Метрики
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)
        rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
        
        # MAPE с защитой от деления на ноль
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        
        # R2 score
        ss_res = np.sum((y_true_filtered - y_pred_filtered) ** 2)
        ss_tot = np.sum((y_true_filtered - np.mean(y_true_filtered)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'model_name': model_name,
            'unified_code': unified_code
        }
        
        # Сохраняем результаты
        key = f"{unified_code}_{model_name}"
        self.evaluation_results[key] = results
        
        return results
    
    def cross_validate(self, data: pd.DataFrame, model, unified_code: str,
                      train_size: float = 0.8) -> Dict:
        """
        Кросс-валидация модели на исторических данных
        
        Args:
            data: Исторические данные
            model: Модель для валидации
            unified_code: Унифицированный код продукта
            train_size: Доля данных для обучения
        
        Returns:
            Словарь с метриками
        """
        if data.empty or len(data) < 10:
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'r2': -np.inf
            }
        
        # Разделение на train/test
        split_idx = int(len(data) * train_size)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        if len(test_data) == 0:
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'r2': -np.inf
            }
        
        # Обучение модели
        try:
            model.fit(train_data, unified_code)
            
            # Прогноз на тестовых данных
            if hasattr(model, 'predict'):
                # Для моделей, требующих будущие признаки
                if hasattr(model, 'predict') and len(test_data) > 0:
                    # Создаем будущие признаки (если нужны)
                    periods = len(test_data)
                    y_pred = model.predict(unified_code, periods=periods)
                else:
                    y_pred = model.predict(unified_code, periods=len(test_data))
            else:
                return {
                    'mae': np.inf,
                    'rmse': np.inf,
                    'mape': np.inf,
                    'r2': -np.inf
                }
            
            y_true = test_data['quantity'].values
            
            # Оценка
            return self.evaluate_model(y_true, y_pred, model.get_model_name(), unified_code)
            
        except Exception as e:
            print(f"Ошибка кросс-валидации для {unified_code}: {e}")
            return {
                'mae': np.inf,
                'rmse': np.inf,
                'mape': np.inf,
                'r2': -np.inf
            }
    
    def select_best_model(self, unified_code: str, metric: str = 'mape') -> str:
        """
        Выбирает лучшую модель для продукта
        
        Args:
            unified_code: Унифицированный код продукта
            metric: Метрика для выбора ('mae', 'rmse', 'mape', 'r2')
        
        Returns:
            Название лучшей модели
        """
        # Фильтруем результаты по продукту
        product_results = {
            k: v for k, v in self.evaluation_results.items()
            if k.startswith(f"{unified_code}_")
        }
        
        if not product_results:
            return None
        
        # Выбираем лучшую модель
        if metric in ['mae', 'rmse', 'mape']:
            # Меньше - лучше
            best_key = min(product_results.keys(), 
                          key=lambda k: product_results[k].get(metric, np.inf))
        else:  # r2
            # Больше - лучше
            best_key = max(product_results.keys(),
                          key=lambda k: product_results[k].get(metric, -np.inf))
        
        return product_results[best_key]['model_name']
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Возвращает сводку по оценке всех моделей
        
        Returns:
            DataFrame с результатами оценки
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        results_list = []
        for key, results in self.evaluation_results.items():
            results_list.append(results)
        
        return pd.DataFrame(results_list)
    
    def get_best_models_summary(self) -> pd.DataFrame:
        """
        Возвращает сводку по лучшим моделям для каждого продукта
        
        Returns:
            DataFrame с лучшими моделями
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        # Группируем по продуктам
        products = set()
        for key in self.evaluation_results.keys():
            unified_code = key.split('_')[0]
            products.add(unified_code)
        
        best_models = []
        for product in products:
            best_model = self.select_best_model(product)
            if best_model:
                # Находим метрики лучшей модели
                key = f"{product}_{best_model}"
                if key in self.evaluation_results:
                    metrics = self.evaluation_results[key].copy()
                    metrics['best_model'] = best_model
                    best_models.append(metrics)
        
        return pd.DataFrame(best_models)


