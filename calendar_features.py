"""
Модуль для создания календарных признаков (праздники, выходные, черная пятница)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays


class CalendarFeatures:
    """Класс для создания календарных признаков"""
    
    def __init__(self, country='RU'):
        """
        Инициализация
        
        Args:
            country: Код страны для праздников (по умолчанию RU - Россия)
        """
        self.country = country
        self.ru_holidays = holidays.Russia(years=range(2020, 2030))
        
        # Даты черной пятницы для Ozon и Wildberries (примерные, нужно уточнить)
        self.ozon_black_friday_dates = [
            pd.Timestamp('2023-11-24'),
            pd.Timestamp('2024-11-29'),
            pd.Timestamp('2025-11-28'),
        ]
        
        self.wb_black_friday_dates = [
            pd.Timestamp('2023-11-24'),
            pd.Timestamp('2024-11-29'),
            pd.Timestamp('2025-11-28'),
        ]
    
    def add_calendar_features(self, df: pd.DataFrame, marketplace: str = 'wb') -> pd.DataFrame:
        """
        Добавляет календарные признаки к DataFrame
        
        Args:
            df: DataFrame с колонкой 'date'
            marketplace: 'wb' или 'ozon'
        
        Returns:
            DataFrame с добавленными признаками
        """
        df = df.copy()
        
        if 'date' not in df.columns:
            return df
        
        # Базовые признаки
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Выходные дни (суббота=5, воскресенье=6)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Праздники
        df['is_holiday'] = df['date'].apply(
            lambda x: 1 if x in self.ru_holidays else 0
        )
        
        # Черная пятница
        if marketplace == 'ozon':
            black_friday_dates = self.ozon_black_friday_dates
        else:
            black_friday_dates = self.wb_black_friday_dates
        
        df['is_black_friday'] = df['date'].apply(
            lambda x: 1 if x in black_friday_dates else 0
        )
        
        # Период вокруг черной пятницы (неделя до и после)
        df['is_black_friday_period'] = 0
        for bf_date in black_friday_dates:
            period_start = bf_date - timedelta(days=7)
            period_end = bf_date + timedelta(days=7)
            mask = (df['date'] >= period_start) & (df['date'] <= period_end)
            df.loc[mask, 'is_black_friday_period'] = 1
        
        # Новогодние праздники (декабрь-январь)
        df['is_new_year_period'] = (
            ((df['month'] == 12) & (df['day'] >= 20)) |
            ((df['month'] == 1) & (df['day'] <= 10))
        ).astype(int)
        
        # Летний период (июнь-август)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Бинарные признаки для месяцев
        for month in range(1, 13):
            df[f'month_{month}'] = (df['month'] == month).astype(int)
        
        # Бинарные признаки для дней недели
        for day in range(7):
            df[f'day_of_week_{day}'] = (df['day_of_week'] == day).astype(int)
        
        return df
    
    def get_future_calendar_features(self, start_date: pd.Timestamp, 
                                     periods: int = 18, 
                                     marketplace: str = 'wb') -> pd.DataFrame:
        """
        Создает календарные признаки для будущих дат
        
        Args:
            start_date: Начальная дата
            periods: Количество месяцев для прогноза
            marketplace: 'wb' или 'ozon'
        
        Returns:
            DataFrame с календарными признаками для будущих дат
        """
        # Создаем даты на 18 месяцев вперед
        dates = pd.date_range(start=start_date, periods=periods * 30, freq='D')
        dates = dates[:periods * 30]  # Ограничиваем примерно 18 месяцами
        
        df = pd.DataFrame({'date': dates})
        df = self.add_calendar_features(df, marketplace=marketplace)
        
        return df
    
    def add_black_friday_dates(self, ozon_dates: list, wb_dates: list):
        """
        Добавляет даты черной пятницы
        
        Args:
            ozon_dates: Список дат черной пятницы для Ozon
            wb_dates: Список дат черной пятницы для Wildberries
        """
        self.ozon_black_friday_dates = [pd.Timestamp(d) for d in ozon_dates]
        self.wb_black_friday_dates = [pd.Timestamp(d) for d in wb_dates]


