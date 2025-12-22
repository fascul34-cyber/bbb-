"""
Основной скрипт для запуска прогнозирования
"""
import argparse
import sys
from pathlib import Path
from .forecaster import SalesForecaster


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Прогнозирование продаж и отгрузок')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Путь к папке с данными')
    parser.add_argument('--marketplace', type=str, choices=['wb', 'ozon', 'both'],
                       default='both', help='Маркетплейс для прогнозирования')
    parser.add_argument('--months', type=int, default=18,
                       help='Количество месяцев для прогноза')
    parser.add_argument('--product', type=str, default=None,
                       help='Унифицированный код продукта (если None - для всех)')
    
    args = parser.parse_args()
    
    # Создаем прогнозировщик
    forecaster = SalesForecaster(
        data_path=args.data_path,
        forecast_months=args.months
    )
    
    # Загружаем данные
    try:
        forecaster.load_data()
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        sys.exit(1)
    
    # Подготавливаем модели
    forecaster.prepare_models()
    
    # Запускаем прогнозирование
    if args.marketplace == 'both':
        marketplaces = ['wb', 'ozon']
    else:
        marketplaces = [args.marketplace]
    
    for marketplace in marketplaces:
        try:
            results = forecaster.run_full_forecast(marketplace=marketplace)
            
            # Выводим краткую статистику
            if 'best_forecast' in results and not results['best_forecast'].empty:
                print(f"\nСтатистика для {marketplace}:")
                print(f"  Продуктов: {results['best_forecast']['unified_code'].nunique()}")
                print(f"  Общий прогноз продаж: {results['best_forecast']['quantity'].sum():.0f}")
                
            if 'shipments' in results and not results['shipments'].empty:
                print(f"  Общая отгрузка: {results['shipments']['shipment'].sum():.0f}")
                print(f"  Складов: {results['shipments']['warehouse'].nunique()}")
            
        except Exception as e:
            print(f"Ошибка прогнозирования для {marketplace}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()


