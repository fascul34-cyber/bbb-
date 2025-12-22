"""
Пример использования системы прогнозирования
"""
from forecasting.forecaster import SalesForecaster
import pandas as pd


def main():
    """Пример использования"""
    
    # 1. Создание прогнозировщика
    forecaster = SalesForecaster(
        data_path='data',  # Путь к папке с данными
        forecast_months=18  # Прогноз на 18 месяцев
    )
    
    # 2. Загрузка данных
    print("Загрузка данных...")
    forecaster.load_data()
    
    # 3. Настройка параметров (опционально)
    # Коэффициент покрытия для отгрузок
    forecaster.shipment_calculator.set_coverage_coefficient(1.5)
    
    # Размеры коробов
    box_sizes = {
        'default': 24,  # По умолчанию 24 штуки
        # Можно указать для конкретных продуктов:
        # 'product_code_1': 56,
        # 'product_code_2': 72,
    }
    forecaster.constraints.set_box_sizes(box_sizes)
    
    # Даты черной пятницы (если нужно обновить)
    ozon_bf = ['2023-11-24', '2024-11-29', '2025-11-28']
    wb_bf = ['2023-11-24', '2024-11-29', '2025-11-28']
    forecaster.calendar_features.add_black_friday_dates(ozon_bf, wb_bf)
    
    # 4. Подготовка моделей
    print("Подготовка моделей...")
    forecaster.prepare_models()
    
    # 5. Запуск прогнозирования для Wildberries
    print("\nПрогнозирование для Wildberries...")
    wb_results = forecaster.run_full_forecast(marketplace='wb', save_results=True)
    
    # 6. Запуск прогнозирования для Ozon
    print("\nПрогнозирование для Ozon...")
    ozon_results = forecaster.run_full_forecast(marketplace='ozon', save_results=True)
    
    # 7. Работа с результатами
    print("\nРезультаты прогнозирования:")
    
    # Лучшие прогнозы
    if 'best_forecast' in wb_results:
        wb_forecast = wb_results['best_forecast']
        print(f"\nWildberries:")
        print(f"  Продуктов: {wb_forecast['unified_code'].nunique()}")
        print(f"  Общий прогноз продаж: {wb_forecast['quantity'].sum():.0f}")
        
        # Сохраняем в файл
        wb_forecast.to_csv('wb_forecast.csv', index=False)
        print("  Сохранено в wb_forecast.csv")
    
    if 'best_forecast' in ozon_results:
        ozon_forecast = ozon_results['best_forecast']
        print(f"\nOzon:")
        print(f"  Продуктов: {ozon_forecast['unified_code'].nunique()}")
        print(f"  Общий прогноз продаж: {ozon_forecast['quantity'].sum():.0f}")
        
        # Сохраняем в файл
        ozon_forecast.to_csv('ozon_forecast.csv', index=False)
        print("  Сохранено в ozon_forecast.csv")
    
    # Отгрузки
    if 'shipments' in wb_results and not wb_results['shipments'].empty:
        wb_shipments = wb_results['shipments']
        print(f"\nОтгрузки Wildberries:")
        print(f"  Общая отгрузка: {wb_shipments['shipment'].sum():.0f}")
        print(f"  Складов: {wb_shipments['warehouse'].nunique()}")
        wb_shipments.to_csv('wb_shipments.csv', index=False)
        print("  Сохранено в wb_shipments.csv")
    
    if 'shipments' in ozon_results and not ozon_results['shipments'].empty:
        ozon_shipments = ozon_results['shipments']
        print(f"\nОтгрузки Ozon:")
        print(f"  Общая отгрузка: {ozon_shipments['shipment'].sum():.0f}")
        print(f"  Складов: {ozon_shipments['warehouse'].nunique()}")
        ozon_shipments.to_csv('ozon_shipments.csv', index=False)
        print("  Сохранено в ozon_shipments.csv")
    
    # Оценка моделей
    if 'evaluation_summary' in wb_results:
        evaluation = wb_results['evaluation_summary']
        if not evaluation.empty:
            print("\nОценка моделей (Wildberries):")
            print(evaluation.head(10))
            evaluation.to_csv('wb_model_evaluation.csv', index=False)
    
    if 'best_models_summary' in wb_results:
        best_models = wb_results['best_models_summary']
        if not best_models.empty:
            print("\nЛучшие модели по продуктам (Wildberries):")
            print(best_models.head(10))
            best_models.to_csv('wb_best_models.csv', index=False)
    
    # 8. Просмотр истории прогнозов
    print("\nПросмотр истории прогнозов...")
    history = forecaster.forecast_manager.get_forecast_history(marketplace='wb')
    if not history.empty:
        print(f"  Найдено {len(history)} записей в истории")
        
        # Сравнение прогнозов для конкретного продукта
        if 'unified_code' in history.columns:
            product = history['unified_code'].iloc[0]
            comparison = forecaster.forecast_manager.compare_forecasts(
                'wb', product
            )
            if not comparison.empty:
                print(f"\nСравнение прогнозов для продукта {product}:")
                print(comparison.head())


if __name__ == '__main__':
    main()


