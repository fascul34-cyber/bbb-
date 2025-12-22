"""
Модуль для загрузки и обработки данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Класс для загрузки и предобработки данных"""
    
    def __init__(self, data_path: str = "data"):
        """
        Инициализация загрузчика данных
        
        Args:
            data_path: Путь к папке с данными
        """
        self.data_path = Path(data_path)
        self.wb_sales = None
        self.ozon_sales = None
        self.wb_stocks = None
        self.ozon_stocks = None
        self.our_stocks = None
        self.withdraw = None
        self.defecture = None
        self.historical_shipments = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Загружает все данные из файлов
        
        Returns:
            Словарь с загруженными данными
        """
        data = {}
        
        # Загрузка продаж
        try:
            self.wb_sales = self._load_sales("wb_sales")
            data['wb_sales'] = self.wb_sales
        except Exception as e:
            print(f"Ошибка загрузки wb_sales: {e}")
            
        try:
            self.ozon_sales = self._load_sales("ozon_sales")
            data['ozon_sales'] = self.ozon_sales
        except Exception as e:
            print(f"Ошибка загрузки ozon_sales: {e}")
        
        # Загрузка остатков
        try:
            self.wb_stocks = self._load_stocks("wb_stocks")
            data['wb_stocks'] = self.wb_stocks
        except Exception as e:
            print(f"Ошибка загрузки wb_stocks: {e}")
            
        try:
            self.ozon_stocks = self._load_stocks("ozon_stocks")
            data['ozon_stocks'] = self.ozon_stocks
        except Exception as e:
            print(f"Ошибка загрузки ozon_stocks: {e}")
        
        # Загрузка остатков на нашем складе
        try:
            self.our_stocks = self._load_our_stocks("our_stocks")
            data['our_stocks'] = self.our_stocks
        except Exception as e:
            print(f"Ошибка загрузки our_stocks: {e}")
        
        # Загрузка списка на вывод
        try:
            self.withdraw = self._load_withdraw("withdraw")
            data['withdraw'] = self.withdraw
        except Exception as e:
            print(f"Ошибка загрузки withdraw: {e}")
        
        # Загрузка дефектуры
        try:
            self.defecture = self._load_defecture("defecture")
            data['defecture'] = self.defecture
        except Exception as e:
            print(f"Ошибка загрузки defecture: {e}")
        
        return data
    
    def _load_sales(self, filename: str) -> pd.DataFrame:
        """Загружает данные о продажах"""
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        # Обработка даты
        if 'Дата' in df.columns:
            df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
        elif 'Годы (Дата)' in df.columns and 'Месяцы (Дата)' in df.columns:
            df['Дата'] = pd.to_datetime(
                df['Годы (Дата)'].astype(str) + '-' + 
                df['Месяцы (Дата)'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
        
        # Переименование колонок для унификации
        column_mapping = {
            'Количество упак.': 'quantity',
            'Унифицированный solo-code': 'unified_code',
            'solo-code': 'solo_code',
            'SKU': 'sku',
            'Дата': 'date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Группировка по дате и унифицированному коду
        if 'quantity' in df.columns and 'unified_code' in df.columns and 'date' in df.columns:
            df = df.groupby(['date', 'unified_code']).agg({
                'quantity': 'sum',
                'sku': 'first',
                'solo_code': 'first'
            }).reset_index()
        
        return df
    
    def _load_stocks(self, filename: str) -> pd.DataFrame:
        """Загружает данные об остатках на маркетплейсах"""
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        # Обработка даты
        if 'Дата' in df.columns:
            df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
        elif 'Годы (Дата)' in df.columns and 'Месяцы (Дата)' in df.columns:
            df['Дата'] = pd.to_datetime(
                df['Годы (Дата)'].astype(str) + '-' + 
                df['Месяцы (Дата)'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
        
        # Переименование колонок
        column_mapping = {
            'Остаток': 'stock',
            'Унифицированный solo-code': 'unified_code',
            'solo-code': 'solo_code',
            'SKU': 'sku',
            'Склад': 'warehouse',
            'Дата': 'date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def _load_our_stocks(self, filename: str) -> pd.DataFrame:
        """Загружает данные об остатках на нашем складе"""
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        # Обработка даты
        if 'Дата' in df.columns:
            df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
        elif 'Годы (Дата)' in df.columns and 'Месяцы (Дата)' in df.columns:
            df['Дата'] = pd.to_datetime(
                df['Годы (Дата)'].astype(str) + '-' + 
                df['Месяцы (Дата)'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
        
        # Переименование колонок
        column_mapping = {
            'Остаток': 'stock',
            'Унифицированный solo-code': 'unified_code',
            'SKU': 'sku',
            'Дата': 'date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def _load_withdraw(self, filename: str) -> pd.DataFrame:
        """Загружает список продуктов на вывод"""
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            return pd.DataFrame(columns=['unified_code', 'sku'])
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        column_mapping = {
            'Унифицированный solo-code': 'unified_code',
            'SKU': 'sku'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df
    
    def _load_defecture(self, filename: str) -> pd.DataFrame:
        """Загружает список продуктов в дефектуре"""
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            return pd.DataFrame(columns=['unified_code', 'sku', 'end_date'])
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        column_mapping = {
            'Унифицированный solo-code': 'unified_code',
            'SKU': 'sku',
            'Дата окончания дефектуры': 'end_date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        if 'end_date' in df.columns:
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        return df
    
    def load_historical_shipments(self, filename: str = "Отгрузки в МП") -> pd.DataFrame:
        """
        Загружает исторические данные отгрузок
        
        Args:
            filename: Имя файла (без расширения)
        
        Returns:
            DataFrame с историческими отгрузками
        """
        file_path = self.data_path / f"{filename}.csv"
        if not file_path.exists():
            file_path = self.data_path / f"{filename}.xlsx"
        
        if not file_path.exists():
            print(f"Файл {filename} не найден")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path) if file_path.suffix == '.csv' else pd.read_excel(file_path)
        
        # Обработка даты
        if 'Дата' in df.columns:
            df['Дата'] = pd.to_datetime(df['Дата'], errors='coerce')
        
        # Переименование колонок
        column_mapping = {
            'Унифицированный solo-code': 'unified_code',
            'Кол-во упаково': 'quantity',
            'Дата': 'date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Группировка по дате и продукту
        if 'quantity' in df.columns and 'unified_code' in df.columns and 'date' in df.columns:
            df = df.groupby(['date', 'unified_code']).agg({
                'quantity': 'sum'
            }).reset_index()
        
        self.historical_shipments = df
        return df
    
    def prepare_sales_data(self, marketplace: str = 'wb') -> pd.DataFrame:
        """
        Подготавливает данные о продажах для прогнозирования
        
        Args:
            marketplace: 'wb' или 'ozon'
        
        Returns:
            DataFrame с подготовленными данными
        """
        if marketplace == 'wb':
            sales_df = self.wb_sales.copy() if self.wb_sales is not None else pd.DataFrame()
        else:
            sales_df = self.ozon_sales.copy() if self.ozon_sales is not None else pd.DataFrame()
        
        if sales_df.empty:
            return pd.DataFrame()
        
        # Создание временного ряда для каждого продукта
        sales_df = sales_df.sort_values(['unified_code', 'date'])
        
        return sales_df
    
    def get_product_list(self) -> pd.DataFrame:
        """Возвращает список всех продуктов с унифицированными кодами"""
        products = set()
        
        if self.wb_sales is not None and 'unified_code' in self.wb_sales.columns:
            products.update(self.wb_sales['unified_code'].unique())
        
        if self.ozon_sales is not None and 'unified_code' in self.ozon_sales.columns:
            products.update(self.ozon_sales['unified_code'].unique())
        
        return pd.DataFrame({'unified_code': list(products)})


