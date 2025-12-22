"""
Модуль для расчета отгрузок по складам маркетплейсов
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ShipmentCalculator:
    """Класс для расчета отгрузок"""
    
    def __init__(self, coverage_coefficient: float = 1.5):
        """
        Инициализация
        
        Args:
            coverage_coefficient: Коэффициент покрытия (сколько нужно иметь на складе относительно продаж)
        """
        self.coverage_coefficient = coverage_coefficient
    
    def calculate_shipments(self, forecast: pd.DataFrame, 
                           stocks: pd.DataFrame,
                           marketplace: str = 'wb') -> pd.DataFrame:
        """
        Рассчитывает отгрузки по складам на основе прогноза продаж
        
        Args:
            forecast: DataFrame с прогнозом продаж (columns: date, unified_code, quantity)
            stocks: DataFrame с остатками по складам (columns: date, warehouse, unified_code, stock)
            marketplace: 'wb' или 'ozon'
        
        Returns:
            DataFrame с расчетными отгрузками по складам
        """
        shipments = []
        
        # Агрегируем прогноз по месяцам для расчета отгрузок
        # (прогноз может быть по дням, но отгрузки обычно планируются помесячно)
        forecast['year_month'] = forecast['date'].dt.to_period('M')
        forecast_monthly = forecast.groupby(['year_month', 'unified_code'])['quantity'].sum().reset_index()
        forecast_monthly['date'] = forecast_monthly['year_month'].dt.to_timestamp()
        
        # Получаем последние остатки по складам
        if not stocks.empty and 'date' in stocks.columns:
            latest_date = stocks['date'].max()
            latest_stocks = stocks[stocks['date'] == latest_date].copy()
        else:
            latest_stocks = stocks.copy()
        
        # Группируем остатки по продуктам и складам
        if not latest_stocks.empty:
            stocks_by_product_warehouse = latest_stocks.groupby(
                ['unified_code', 'warehouse']
            )['stock'].sum().reset_index()
        else:
            stocks_by_product_warehouse = pd.DataFrame(columns=['unified_code', 'warehouse', 'stock'])
        
        for _, row in forecast_monthly.iterrows():
            date = row['date']
            unified_code = row['unified_code']
            forecasted_sales_monthly = row['quantity']
            
            # Получаем остатки по складам для этого продукта
            product_stocks = stocks_by_product_warehouse[
                stocks_by_product_warehouse['unified_code'] == unified_code
            ].copy()
            
            if product_stocks.empty:
                # Если нет остатков, пропускаем (или можно добавить логику распределения)
                continue
            
            # Рассчитываем необходимый остаток на складе для месяца
            # coverage_coefficient определяет, сколько нужно иметь на складе
            required_stock_total = forecasted_sales_monthly * self.coverage_coefficient
            
            # Текущий остаток по всем складам
            current_stock_total = product_stocks['stock'].sum()
            
            # Если общий остаток меньше требуемого, нужна отгрузка
            if current_stock_total < required_stock_total:
                total_shipment_needed = required_stock_total - current_stock_total
                
                # Распределяем отгрузку между складами пропорционально их текущим остаткам
                # или равномерно, если остатков нет
                if current_stock_total > 0:
                    # Пропорциональное распределение
                    product_stocks['shipment'] = (
                        total_shipment_needed * product_stocks['stock'] / current_stock_total
                    )
                else:
                    # Равномерное распределение
                    n_warehouses = len(product_stocks)
                    product_stocks['shipment'] = total_shipment_needed / n_warehouses
                
                # Добавляем информацию об отгрузках
                for _, stock_row in product_stocks.iterrows():
                    if stock_row['shipment'] > 0:
                        shipments.append({
                            'date': date,
                            'warehouse': stock_row['warehouse'],
                            'unified_code': unified_code,
                            'forecasted_sales': forecasted_sales_monthly,
                            'current_stock': stock_row['stock'],
                            'required_stock': required_stock_total,
                            'shipment': stock_row['shipment']
                        })
            else:
                # Если остатков достаточно, но на каком-то складе может быть мало,
                # можно добавить логику перераспределения
                # Пока просто проверяем, что на каждом складе достаточно
                for _, stock_row in product_stocks.iterrows():
                    warehouse = stock_row['warehouse']
                    current_stock = stock_row['stock']
                    
                    # Если на складе остаток меньше среднего требуемого на склад
                    avg_required_per_warehouse = required_stock_total / len(product_stocks)
                    if current_stock < avg_required_per_warehouse:
                        shipment = avg_required_per_warehouse - current_stock
                        shipments.append({
                            'date': date,
                            'warehouse': warehouse,
                            'unified_code': unified_code,
                            'forecasted_sales': forecasted_sales_monthly,
                            'current_stock': current_stock,
                            'required_stock': avg_required_per_warehouse,
                            'shipment': shipment
                        })
        
        if not shipments:
            return pd.DataFrame(columns=['date', 'warehouse', 'unified_code', 'shipment'])
        
        result = pd.DataFrame(shipments)
        return result
    
    def analyze_shipment_calculation(self, historical_sales: pd.DataFrame,
                                   historical_stocks: pd.DataFrame,
                                   historical_shipments: pd.DataFrame = None) -> Dict:
        """
        Анализирует исторические данные для понимания логики расчета отгрузок
        
        Args:
            historical_sales: Исторические продажи
            historical_stocks: Исторические остатки
            historical_shipments: Исторические отгрузки (если есть)
        
        Returns:
            Словарь с анализом
        """
        analysis = {}
        
        if historical_sales.empty or historical_stocks.empty:
            return analysis
        
        # Анализ связи между продажами и остатками
        # Группируем продажи по продуктам и датам
        sales_grouped = historical_sales.groupby(['date', 'unified_code'])['quantity'].sum().reset_index()
        
        # Анализируем для нескольких продуктов
        products = sales_grouped['unified_code'].unique()[:10]  # Берем первые 10
        
        coverage_ratios = []
        
        for product in products:
            product_sales = sales_grouped[sales_grouped['unified_code'] == product]
            product_stocks = historical_stocks[historical_stocks['unified_code'] == product]
            
            if product_sales.empty or product_stocks.empty:
                continue
            
            # Находим общие даты
            common_dates = set(product_sales['date']) & set(product_stocks['date'])
            
            for date in list(common_dates)[:5]:  # Берем первые 5 дат
                sales = product_sales[product_sales['date'] == date]['quantity'].sum()
                stocks = product_stocks[product_stocks['date'] == date]['stock'].sum()
                
                if sales > 0:
                    ratio = stocks / sales
                    coverage_ratios.append(ratio)
        
        if coverage_ratios:
            analysis['avg_coverage_ratio'] = np.mean(coverage_ratios)
            analysis['median_coverage_ratio'] = np.median(coverage_ratios)
            analysis['min_coverage_ratio'] = np.min(coverage_ratios)
            analysis['max_coverage_ratio'] = np.max(coverage_ratios)
        
        # Если есть исторические отгрузки, анализируем их
        if historical_shipments is not None and not historical_shipments.empty:
            # Анализ связи отгрузок с продажами и остатками
            # Группируем отгрузки по датам
            shipments_grouped = historical_shipments.groupby(['date', 'unified_code'])['quantity'].sum().reset_index()
            
            # Находим общие даты для анализа
            common_dates = set(sales_grouped['date']) & set(shipments_grouped['date'])
            
            shipment_ratios = []
            coverage_from_shipments = []
            
            for date in list(common_dates)[:20]:  # Берем первые 20 дат
                date_sales = sales_grouped[sales_grouped['date'] == date]
                date_shipments = shipments_grouped[shipments_grouped['date'] == date]
                
                # Находим общие продукты
                common_products = set(date_sales['unified_code']) & set(date_shipments['unified_code'])
                
                for product in common_products:
                    sales = date_sales[date_sales['unified_code'] == product]['quantity'].sum()
                    shipments = date_shipments[date_shipments['unified_code'] == product]['quantity'].sum()
                    
                    if sales > 0:
                        # Коэффициент отгрузки к продажам
                        ratio = shipments / sales
                        shipment_ratios.append(ratio)
                    
                    # Анализ покрытия через отгрузки
                    # Если была отгрузка, значит был недостаток остатков
                    if shipments > 0:
                        # Получаем остатки на эту дату
                        date_stocks = historical_stocks[
                            (historical_stocks['date'] == date) & 
                            (historical_stocks['unified_code'] == product)
                        ]
                        if not date_stocks.empty:
                            total_stock = date_stocks['stock'].sum()
                            if sales > 0:
                                coverage = (total_stock + shipments) / sales
                                coverage_from_shipments.append(coverage)
            
            if shipment_ratios:
                analysis['avg_shipment_ratio'] = np.mean(shipment_ratios)
                analysis['median_shipment_ratio'] = np.median(shipment_ratios)
                analysis['min_shipment_ratio'] = np.min(shipment_ratios)
                analysis['max_shipment_ratio'] = np.max(shipment_ratios)
            
            if coverage_from_shipments:
                analysis['avg_coverage_from_shipments'] = np.mean(coverage_from_shipments)
                analysis['median_coverage_from_shipments'] = np.median(coverage_from_shipments)
        
        return analysis
    
    def set_coverage_coefficient(self, coefficient: float):
        """
        Устанавливает коэффициент покрытия
        
        Args:
            coefficient: Новый коэффициент покрытия
        """
        self.coverage_coefficient = coefficient

