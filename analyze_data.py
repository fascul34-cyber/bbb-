"""
Скрипт для анализа исходных данных и понимания логики расчета отгрузок
"""
import pandas as pd
import numpy as np
from pathlib import Path
from .data_loader import DataLoader
from .shipment_calculator import ShipmentCalculator


def analyze_shipment_logic(data_path: str = "data"):
    """
    Анализирует исторические данные для понимания логики расчета отгрузок
    
    Args:
        data_path: Путь к данным
    """
    print("Анализ данных для понимания логики расчета отгрузок\n")
    
    # Загрузка данных
    loader = DataLoader(data_path)
    data = loader.load_all_data()
    
    shipment_calc = ShipmentCalculator()
    
    # Анализ для Wildberries
    if 'wb_sales' in data and 'wb_stocks' in data:
        print("=" * 60)
        print("АНАЛИЗ WILDBERRIES")
        print("=" * 60)
        
        wb_sales = data['wb_sales']
        wb_stocks = data['wb_stocks']
        
        print(f"\nПродажи:")
        print(f"  Период: {wb_sales['date'].min()} - {wb_sales['date'].max()}")
        print(f"  Уникальных продуктов: {wb_sales['unified_code'].nunique()}")
        print(f"  Всего записей: {len(wb_sales)}")
        print(f"  Средние продажи в день: {wb_sales.groupby('date')['quantity'].sum().mean():.2f}")
        
        print(f"\nОстатки:")
        print(f"  Период: {wb_stocks['date'].min()} - {wb_stocks['date'].max()}")
        print(f"  Уникальных складов: {wb_stocks['warehouse'].nunique()}")
        print(f"  Уникальных продуктов: {wb_stocks['unified_code'].nunique()}")
        print(f"  Всего записей: {len(wb_stocks)}")
        
        # Анализ связи продаж и остатков
        analysis = shipment_calc.analyze_shipment_calculation(
            wb_sales, wb_stocks
        )
        
        if analysis:
            print(f"\nАнализ коэффициента покрытия:")
            print(f"  Средний коэффициент: {analysis.get('avg_coverage_ratio', 0):.2f}")
            print(f"  Медианный коэффициент: {analysis.get('median_coverage_ratio', 0):.2f}")
            print(f"  Минимальный коэффициент: {analysis.get('min_coverage_ratio', 0):.2f}")
            print(f"  Максимальный коэффициент: {analysis.get('max_coverage_ratio', 0):.2f}")
        
        # Анализ по продуктам
        print(f"\nТоп-10 продуктов по продажам:")
        top_products = wb_sales.groupby('unified_code')['quantity'].sum().sort_values(ascending=False).head(10)
        for product, sales in top_products.items():
            print(f"  {product}: {sales:.0f}")
        
        # Анализ по складам
        if 'warehouse' in wb_stocks.columns:
            print(f"\nРаспределение остатков по складам:")
            warehouse_stocks = wb_stocks.groupby('warehouse')['stock'].sum().sort_values(ascending=False)
            for warehouse, stock in warehouse_stocks.items():
                print(f"  {warehouse}: {stock:.0f}")
    
    # Анализ для Ozon
    if 'ozon_sales' in data and 'ozon_stocks' in data:
        print("\n" + "=" * 60)
        print("АНАЛИЗ OZON")
        print("=" * 60)
        
        ozon_sales = data['ozon_sales']
        ozon_stocks = data['ozon_stocks']
        
        print(f"\nПродажи:")
        print(f"  Период: {ozon_sales['date'].min()} - {ozon_sales['date'].max()}")
        print(f"  Уникальных продуктов: {ozon_sales['unified_code'].nunique()}")
        print(f"  Всего записей: {len(ozon_sales)}")
        print(f"  Средние продажи в день: {ozon_sales.groupby('date')['quantity'].sum().mean():.2f}")
        
        print(f"\nОстатки:")
        print(f"  Период: {ozon_stocks['date'].min()} - {ozon_stocks['date'].max()}")
        print(f"  Уникальных складов: {ozon_stocks['warehouse'].nunique()}")
        print(f"  Уникальных продуктов: {ozon_stocks['unified_code'].nunique()}")
        print(f"  Всего записей: {len(ozon_stocks)}")
        
        # Анализ связи продаж и остатков
        analysis = shipment_calc.analyze_shipment_calculation(
            ozon_sales, ozon_stocks
        )
        
        if analysis:
            print(f"\nАнализ коэффициента покрытия:")
            print(f"  Средний коэффициент: {analysis.get('avg_coverage_ratio', 0):.2f}")
            print(f"  Медианный коэффициент: {analysis.get('median_coverage_ratio', 0):.2f}")
            print(f"  Минимальный коэффициент: {analysis.get('min_coverage_ratio', 0):.2f}")
            print(f"  Максимальный коэффициент: {analysis.get('max_coverage_ratio', 0):.2f}")
        
        # Анализ по продуктам
        print(f"\nТоп-10 продуктов по продажам:")
        top_products = ozon_sales.groupby('unified_code')['quantity'].sum().sort_values(ascending=False).head(10)
        for product, sales in top_products.items():
            print(f"  {product}: {sales:.0f}")
        
        # Анализ по складам
        if 'warehouse' in ozon_stocks.columns:
            print(f"\nРаспределение остатков по складам:")
            warehouse_stocks = ozon_stocks.groupby('warehouse')['stock'].sum().sort_values(ascending=False)
            for warehouse, stock in warehouse_stocks.items():
                print(f"  {warehouse}: {stock:.0f}")
    
    # Анализ остатков на нашем складе
    if 'our_stocks' in data:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ОСТАТКОВ НА НАШЕМ СКЛАДЕ")
        print("=" * 60)
        
        our_stocks = data['our_stocks']
        print(f"\nОстатки:")
        print(f"  Период: {our_stocks['date'].min()} - {our_stocks['date'].max()}")
        print(f"  Уникальных продуктов: {our_stocks['unified_code'].nunique()}")
        print(f"  Всего записей: {len(our_stocks)}")
        print(f"  Средний остаток: {our_stocks['stock'].mean():.2f}")
        print(f"  Общий остаток (последняя дата): {our_stocks.groupby('date')['stock'].sum().iloc[-1]:.0f}")
    
    # Анализ продуктов на вывод
    if 'withdraw' in data:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ПРОДУКТОВ НА ВЫВОД")
        print("=" * 60)
        
        withdraw = data['withdraw']
        print(f"\nПродуктов на вывод: {len(withdraw)}")
        if not withdraw.empty:
            print("  Примеры:")
            print(withdraw.head(10))
    
    # Анализ дефектуры
    if 'defecture' in data:
        print("\n" + "=" * 60)
        print("АНАЛИЗ ДЕФЕКТУРЫ")
        print("=" * 60)
        
        defecture = data['defecture']
        print(f"\nПродуктов в дефектуре: {len(defecture)}")
        if not defecture.empty and 'end_date' in defecture.columns:
            print(f"  Период дефектуры: {defecture['end_date'].min()} - {defecture['end_date'].max()}")
            print("  Примеры:")
            print(defecture.head(10))


if __name__ == '__main__':
    analyze_shipment_logic()

