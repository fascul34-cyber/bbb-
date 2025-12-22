"""
Модуль для применения ограничений к прогнозам и отгрузкам
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class Constraints:
    """Класс для применения ограничений"""
    
    def __init__(self, box_sizes: Dict[str, int] = None):
        """
        Инициализация
        
        Args:
            box_sizes: Словарь {unified_code: размер_короба} или общий размер короба
        """
        self.box_sizes = box_sizes if box_sizes else {}
        self.default_box_size = 24  # По умолчанию 24 штуки в коробе
    
    def apply_withdraw_constraints(self, forecast: pd.DataFrame,
                                  withdraw_list: pd.DataFrame,
                                  wb_stocks: pd.DataFrame,
                                  ozon_stocks: pd.DataFrame,
                                  our_stocks: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет ограничения для продуктов на вывод
        
        Args:
            forecast: DataFrame с прогнозом
            withdraw_list: Список продуктов на вывод
            wb_stocks: Остатки на Wildberries
            ozon_stocks: Остатки на Ozon
            our_stocks: Остатки на нашем складе
        
        Returns:
            DataFrame с примененными ограничениями
        """
        forecast = forecast.copy()
        
        if withdraw_list.empty:
            return forecast
        
        # Получаем список продуктов на вывод
        withdraw_codes = set(withdraw_list['unified_code'].unique())
        
        # Получаем последние остатки
        latest_wb_stocks = self._get_latest_stocks(wb_stocks)
        latest_ozon_stocks = self._get_latest_stocks(ozon_stocks)
        latest_our_stocks = self._get_latest_stocks(our_stocks)
        
        # Применяем ограничения
        for unified_code in withdraw_codes:
            if unified_code not in forecast['unified_code'].values:
                continue
            
            # Суммируем остатки
            wb_total = latest_wb_stocks.get(unified_code, 0)
            ozon_total = latest_ozon_stocks.get(unified_code, 0)
            our_total = latest_our_stocks.get(unified_code, 0)
            
            total_stock = wb_total + ozon_total + our_total
            
            # Если остатков нет, обнуляем прогноз
            if total_stock <= 0:
                forecast.loc[forecast['unified_code'] == unified_code, 'quantity'] = 0
        
        return forecast
    
    def apply_defecture_constraints(self, forecast: pd.DataFrame,
                                   defecture_list: pd.DataFrame,
                                   wb_stocks: pd.DataFrame,
                                   ozon_stocks: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет ограничения для продуктов в дефектуре
        
        Args:
            forecast: DataFrame с прогнозом
            defecture_list: Список продуктов в дефектуре
            wb_stocks: Остатки на Wildberries
            ozon_stocks: Остатки на Ozon
        
        Returns:
            DataFrame с примененными ограничениями
        """
        forecast = forecast.copy()
        
        if defecture_list.empty:
            return forecast
        
        # Получаем последние остатки на маркетплейсах
        latest_wb_stocks = self._get_latest_stocks(wb_stocks)
        latest_ozon_stocks = self._get_latest_stocks(ozon_stocks)
        
        # Применяем ограничения
        for _, defecture_row in defecture_list.iterrows():
            unified_code = defecture_row['unified_code']
            end_date = defecture_row.get('end_date', pd.Timestamp.max)
            
            if unified_code not in forecast['unified_code'].values:
                continue
            
            # Проверяем дату окончания дефектуры
            future_dates = forecast[
                (forecast['unified_code'] == unified_code) & 
                (forecast['date'] > end_date)
            ]
            
            if len(future_dates) == 0:
                # Если дефектура еще не закончилась, проверяем остатки
                wb_total = latest_wb_stocks.get(unified_code, 0)
                ozon_total = latest_ozon_stocks.get(unified_code, 0)
                
                total_stock = wb_total + ozon_total
                
                # Если остатков нет, обнуляем прогноз
                if total_stock <= 0:
                    forecast.loc[forecast['unified_code'] == unified_code, 'quantity'] = 0
        
        return forecast
    
    def apply_box_constraints(self, shipments: pd.DataFrame,
                             box_sizes: Dict[str, int] = None) -> pd.DataFrame:
        """
        Округляет отгрузки до размера короба
        
        Args:
            shipments: DataFrame с отгрузками
            box_sizes: Словарь {unified_code: размер_короба} или общий размер
        
        Returns:
            DataFrame с округленными отгрузками
        """
        shipments = shipments.copy()
        
        if shipments.empty:
            return shipments
        
        if box_sizes is None:
            box_sizes = self.box_sizes
        
        # Округляем отгрузки
        for idx, row in shipments.iterrows():
            unified_code = row['unified_code']
            shipment = row['shipment']
            
            # Определяем размер короба
            if box_sizes and unified_code in box_sizes:
                box_size = box_sizes[unified_code]
            elif isinstance(box_sizes, dict) and 'default' in box_sizes:
                box_size = box_sizes['default']
            else:
                box_size = self.default_box_size
            
            # Округляем вверх до ближайшего короба
            boxes = np.ceil(shipment / box_size)
            rounded_shipment = boxes * box_size
            
            shipments.loc[idx, 'shipment'] = rounded_shipment
            shipments.loc[idx, 'boxes'] = boxes
        
        return shipments
    
    def apply_shipment_withdraw_constraints(self, shipments: pd.DataFrame,
                                          withdraw_list: pd.DataFrame) -> pd.DataFrame:
        """
        Обнуляет отгрузки для продуктов на вывод
        
        Args:
            shipments: DataFrame с отгрузками
            withdraw_list: Список продуктов на вывод
        
        Returns:
            DataFrame с примененными ограничениями
        """
        shipments = shipments.copy()
        
        if withdraw_list.empty:
            return shipments
        
        withdraw_codes = set(withdraw_list['unified_code'].unique())
        shipments.loc[shipments['unified_code'].isin(withdraw_codes), 'shipment'] = 0
        
        return shipments
    
    def apply_shipment_defecture_constraints(self, shipments: pd.DataFrame,
                                           defecture_list: pd.DataFrame) -> pd.DataFrame:
        """
        Обнуляет отгрузки для продуктов в дефектуре
        
        Args:
            shipments: DataFrame с отгрузками
            defecture_list: Список продуктов в дефектуре
        
        Returns:
            DataFrame с примененными ограничениями
        """
        shipments = shipments.copy()
        
        if defecture_list.empty:
            return shipments
        
        defecture_codes = set(defecture_list['unified_code'].unique())
        shipments.loc[shipments['unified_code'].isin(defecture_codes), 'shipment'] = 0
        
        return shipments
    
    def _get_latest_stocks(self, stocks: pd.DataFrame) -> Dict[str, float]:
        """
        Получает последние остатки по продуктам
        
        Args:
            stocks: DataFrame с остатками
        
        Returns:
            Словарь {unified_code: stock}
        """
        if stocks.empty:
            return {}
        
        if 'date' in stocks.columns:
            latest_date = stocks['date'].max()
            latest_stocks = stocks[stocks['date'] == latest_date]
        else:
            latest_stocks = stocks
        
        if 'unified_code' in latest_stocks.columns and 'stock' in latest_stocks.columns:
            result = latest_stocks.groupby('unified_code')['stock'].sum().to_dict()
            return result
        
        return {}
    
    def set_box_sizes(self, box_sizes: Dict[str, int]):
        """
        Устанавливает размеры коробов
        
        Args:
            box_sizes: Словарь {unified_code: размер_короба}
        """
        self.box_sizes = box_sizes


