"""
Главный модуль для прогнозирования продаж и отгрузок
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

from .data_loader import DataLoader
from .calendar_features import CalendarFeatures
from .models.baseline import BaselineModel
from .models.linear_regression import LinearRegressionModel, BinaryLinearRegressionModel
from .models.arima import ARIMAModel, SARIMAXModel
from .models.prophet import ProphetModel
from .model_evaluator import ModelEvaluator
from .shipment_calculator import ShipmentCalculator
from .constraints import Constraints
from .forecast_manager import ForecastManager


class SalesForecaster:
    """Главный класс для прогнозирования продаж и отгрузок"""
    
    def __init__(self, data_path: str = "data", forecast_months: int = 18):
        """
        Инициализация
        
        Args:
            data_path: Путь к данным
            forecast_months: Количество месяцев для прогноза
        """
        self.data_path = data_path
        self.forecast_months = forecast_months
        
        # Инициализация компонентов
        self.data_loader = DataLoader(data_path)
        self.calendar_features = CalendarFeatures()
        self.evaluator = ModelEvaluator()
        self.shipment_calculator = ShipmentCalculator()
        self.constraints = Constraints()
        self.forecast_manager = ForecastManager()
        
        # Данные
        self.data = {}
        self.models = {}
        
    def load_data(self):
        """Загружает все данные"""
        print("Загрузка данных...")
        self.data = self.data_loader.load_all_data()
        print("Данные загружены")
    
    def prepare_models(self):
        """Подготавливает модели для прогнозирования"""
        print("Подготовка моделей...")
        
        # Baseline модели
        self.models['baseline_mean'] = BaselineModel(method='mean')
        self.models['baseline_median'] = BaselineModel(method='median')
        self.models['baseline_last'] = BaselineModel(method='last')
        
        # Линейная регрессия
        self.models['linear_regression'] = LinearRegressionModel(use_feature_selection=True)
        self.models['binary_linear_regression'] = BinaryLinearRegressionModel()
        
        # ARIMA модели
        try:
            self.models['arima'] = ARIMAModel(order=(1, 1, 1))
            self.models['sarima'] = ARIMAModel(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            )
            self.models['sarimax'] = SARIMAXModel(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12)
            )
        except ImportError:
            print("ARIMA модели недоступны (statsmodels не установлен)")
        
        # Prophet
        try:
            # Подготовка праздников для Prophet
            holidays_df = self._prepare_prophet_holidays()
            self.models['prophet'] = ProphetModel(holidays=holidays_df)
        except ImportError:
            print("Prophet модель недоступна (prophet не установлен)")
        
        print(f"Подготовлено {len(self.models)} моделей")
    
    def _prepare_prophet_holidays(self) -> pd.DataFrame:
        """Подготавливает праздники для Prophet"""
        holidays_list = []
        
        if not HOLIDAYS_AVAILABLE:
            return None
        
        # Российские праздники
        ru_holidays = holidays.Russia(years=range(2020, 2030))
        for date, name in ru_holidays.items():
            holidays_list.append({
                'ds': pd.Timestamp(date),
                'holiday': name
            })
        
        # Черная пятница
        for bf_date in self.calendar_features.ozon_black_friday_dates:
            holidays_list.append({
                'ds': bf_date,
                'holiday': 'Black Friday Ozon'
            })
        
        for bf_date in self.calendar_features.wb_black_friday_dates:
            holidays_list.append({
                'ds': bf_date,
                'holiday': 'Black Friday WB'
            })
        
        return pd.DataFrame(holidays_list) if holidays_list else None
    
    def forecast_sales(self, marketplace: str = 'wb', 
                      unified_code: str = None,
                      evaluate: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Прогнозирует продажи для маркетплейса
        
        Args:
            marketplace: 'wb' или 'ozon'
            unified_code: Код продукта (если None - для всех продуктов)
            evaluate: Оценивать ли модели
        
        Returns:
            Словарь {model_name: forecast_dataframe}
        """
        print(f"Прогнозирование продаж для {marketplace}...")
        
        # Подготовка данных
        sales_data = self.data_loader.prepare_sales_data(marketplace)
        if sales_data.empty:
            print(f"Нет данных о продажах для {marketplace}")
            return {}
        
        # Получаем список продуктов
        if unified_code:
            products = [unified_code]
        else:
            products = sales_data['unified_code'].unique()
        
        all_forecasts = {}
        
        for product in products:
            print(f"  Прогнозирование для продукта {product}...")
            
            # Данные продукта
            product_data = sales_data[sales_data['unified_code'] == product].copy()
            if product_data.empty:
                continue
            
            # Добавляем календарные признаки
            product_data = self.calendar_features.add_calendar_features(
                product_data, marketplace=marketplace
            )
            
            # Создаем будущие даты
            last_date = product_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=self.forecast_months * 30,
                freq='D'
            )[:self.forecast_months * 30]
            
            future_df = pd.DataFrame({'date': future_dates})
            future_df = self.calendar_features.add_calendar_features(
                future_df, marketplace=marketplace
            )
            future_df['unified_code'] = product
            
            # Прогнозирование каждой моделью
            product_forecasts = {}
            
            for model_name, model in self.models.items():
                try:
                    # Обучение модели
                    if model_name in ['linear_regression', 'binary_linear_regression']:
                        model.fit(product_data, product)
                        forecast_values = model.predict(product, future_df, periods=self.forecast_months)
                    elif model_name == 'sarimax':
                        # Для SARIMAX нужны экзогенные переменные
                        exog_cols = [col for col in product_data.columns 
                                   if col.startswith('is_') or col.startswith('month_')]
                        if exog_cols:
                            model.fit(product_data, product, exog_columns=exog_cols)
                            future_exog = future_df[exog_cols]
                            forecast_values = model.predict(product, future_exog, periods=self.forecast_months)
                        else:
                            forecast_values = np.zeros(self.forecast_months)
                    else:
                        model.fit(product_data, product)
                        forecast_values = model.predict(product, periods=self.forecast_months)
                    
                    # Создаем DataFrame с прогнозом
                    forecast_df = pd.DataFrame({
                        'date': future_dates[:len(forecast_values)],
                        'unified_code': product,
                        'quantity': forecast_values,
                        'model_name': model_name
                    })
                    
                    product_forecasts[model_name] = forecast_df
                    
                    # Оценка модели (если нужно)
                    if evaluate and len(product_data) > 10:
                        try:
                            self.evaluator.cross_validate(
                                product_data, model, product
                            )
                        except Exception as e:
                            print(f"    Ошибка оценки модели {model_name}: {e}")
                    
                except Exception as e:
                    print(f"    Ошибка прогнозирования моделью {model_name} для {product}: {e}")
                    continue
            
            # Сохраняем прогнозы
            for model_name, forecast_df in product_forecasts.items():
                if model_name not in all_forecasts:
                    all_forecasts[model_name] = []
                all_forecasts[model_name].append(forecast_df)
        
        # Объединяем прогнозы
        result = {}
        for model_name, forecasts_list in all_forecasts.items():
            if forecasts_list:
                result[model_name] = pd.concat(forecasts_list, ignore_index=True)
        
        return result
    
    def select_best_forecasts(self, forecasts: Dict[str, pd.DataFrame],
                             marketplace: str) -> pd.DataFrame:
        """
        Выбирает лучшие прогнозы для каждого продукта
        
        Args:
            forecasts: Словарь с прогнозами всех моделей
            marketplace: Маркетплейс
        
        Returns:
            DataFrame с лучшими прогнозами
        """
        print("Выбор лучших прогнозов...")
        
        best_forecasts = []
        
        # Получаем все продукты
        all_products = set()
        for forecast_df in forecasts.values():
            all_products.update(forecast_df['unified_code'].unique())
        
        for product in all_products:
            # Выбираем лучшую модель для продукта
            best_model = self.evaluator.select_best_model(product, metric='mape')
            
            if best_model and best_model in forecasts:
                product_forecast = forecasts[best_model][
                    forecasts[best_model]['unified_code'] == product
                ].copy()
                product_forecast['selected_model'] = best_model
                best_forecasts.append(product_forecast)
        
        if not best_forecasts:
            return pd.DataFrame()
        
        return pd.concat(best_forecasts, ignore_index=True)
    
    def apply_constraints_to_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет ограничения к прогнозу
        
        Args:
            forecast: DataFrame с прогнозом
        
        Returns:
            DataFrame с примененными ограничениями
        """
        print("Применение ограничений к прогнозу...")
        
        # Ограничения для продуктов на вывод
        if 'withdraw' in self.data and not self.data['withdraw'].empty:
            forecast = self.constraints.apply_withdraw_constraints(
                forecast,
                self.data['withdraw'],
                self.data.get('wb_stocks', pd.DataFrame()),
                self.data.get('ozon_stocks', pd.DataFrame()),
                self.data.get('our_stocks', pd.DataFrame())
            )
        
        # Ограничения для продуктов в дефектуре
        if 'defecture' in self.data and not self.data['defecture'].empty:
            forecast = self.constraints.apply_defecture_constraints(
                forecast,
                self.data['defecture'],
                self.data.get('wb_stocks', pd.DataFrame()),
                self.data.get('ozon_stocks', pd.DataFrame())
            )
        
        return forecast
    
    def calculate_shipments(self, forecast: pd.DataFrame,
                           marketplace: str = 'wb') -> pd.DataFrame:
        """
        Рассчитывает отгрузки на основе прогноза
        
        Args:
            forecast: DataFrame с прогнозом продаж
            marketplace: Маркетплейс
        
        Returns:
            DataFrame с отгрузками
        """
        print(f"Расчет отгрузок для {marketplace}...")
        
        stocks = self.data.get(f'{marketplace}_stocks', pd.DataFrame())
        if stocks.empty:
            print(f"Нет данных об остатках для {marketplace}")
            return pd.DataFrame()
        
        # Агрегируем прогноз по месяцам для расчета отгрузок
        forecast_for_shipment = forecast.copy()
        if 'date' in forecast_for_shipment.columns:
            forecast_for_shipment['date'] = pd.to_datetime(forecast_for_shipment['date'])
            # Создаем месячную агрегацию
            forecast_for_shipment['year_month'] = forecast_for_shipment['date'].dt.to_period('M')
            forecast_monthly = forecast_for_shipment.groupby(
                ['year_month', 'unified_code']
            ).agg({
                'quantity': 'sum'
            }).reset_index()
            forecast_monthly['date'] = forecast_monthly['year_month'].dt.to_timestamp()
            forecast_for_shipment = forecast_monthly[['date', 'unified_code', 'quantity']]
        
        shipments = self.shipment_calculator.calculate_shipments(
            forecast_for_shipment, stocks, marketplace=marketplace
        )
        
        # Применяем ограничения к отгрузкам
        if 'withdraw' in self.data and not self.data['withdraw'].empty:
            shipments = self.constraints.apply_shipment_withdraw_constraints(
                shipments, self.data['withdraw']
            )
        
        if 'defecture' in self.data and not self.data['defecture'].empty:
            shipments = self.constraints.apply_shipment_defecture_constraints(
                shipments, self.data['defecture']
            )
        
        # Округляем до коробов
        shipments = self.constraints.apply_box_constraints(shipments)
        
        return shipments
    
    def run_full_forecast(self, marketplace: str = 'wb',
                         save_results: bool = True) -> Dict:
        """
        Запускает полный цикл прогнозирования
        
        Args:
            marketplace: Маркетплейс
            save_results: Сохранять ли результаты
        
        Returns:
            Словарь с результатами
        """
        print(f"\n{'='*50}")
        print(f"Запуск полного прогнозирования для {marketplace}")
        print(f"{'='*50}\n")
        
        # Прогнозирование всеми моделями
        all_forecasts = self.forecast_sales(marketplace=marketplace, evaluate=True)
        
        if not all_forecasts:
            print("Не удалось создать прогнозы")
            return {}
        
        # Выбор лучших прогнозов
        best_forecast = self.select_best_forecasts(all_forecasts, marketplace)
        
        if best_forecast.empty:
            print("Не удалось выбрать лучшие прогнозы")
            return {}
        
        # Применение ограничений
        best_forecast = self.apply_constraints_to_forecast(best_forecast)
        
        # Расчет отгрузок
        shipments = self.calculate_shipments(best_forecast, marketplace=marketplace)
        
        # Сохранение результатов
        if save_results:
            for model_name, forecast_df in all_forecasts.items():
                self.forecast_manager.save_forecast(
                    forecast_df,
                    model_name=model_name,
                    marketplace=marketplace,
                    metadata={
                        'forecast_months': self.forecast_months,
                        'products_count': forecast_df['unified_code'].nunique()
                    }
                )
            
            # Сохраняем лучший прогноз
            self.forecast_manager.save_forecast(
                best_forecast,
                model_name='best',
                marketplace=marketplace,
                metadata={
                    'forecast_months': self.forecast_months,
                    'products_count': best_forecast['unified_code'].nunique(),
                    'selected_models': best_forecast['selected_model'].unique().tolist()
                }
            )
        
        results = {
            'all_forecasts': all_forecasts,
            'best_forecast': best_forecast,
            'shipments': shipments,
            'evaluation_summary': self.evaluator.get_evaluation_summary(),
            'best_models_summary': self.evaluator.get_best_models_summary()
        }
        
        print(f"\n{'='*50}")
        print("Прогнозирование завершено")
        print(f"{'='*50}\n")
        
        return results

