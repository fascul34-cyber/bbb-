# Система прогнозирования продаж и отгрузок для маркетплейсов

Система для прогнозирования продаж на маркетплейсах (Wildberries и Ozon) и расчета отгрузок на 18 месяцев вперед.

## Возможности

- **Множественные модели прогнозирования:**
  - Baseline (среднее, медиана, последнее значение)
  - Линейная регрессия с подбором фичей
  - Линейная регрессия с бинарными признаками
  - ARIMA/SARIMA/SARIMAX
  - Prophet

- **Учет календарных факторов:**
  - Праздники
  - Выходные дни
  - Черная пятница (Ozon и Wildberries)
  - Сезонность

- **Оценка качества моделей:**
  - Метрики: MAE, RMSE, MAPE, R²
  - Автоматический выбор лучшей модели для каждого продукта

- **Ограничения:**
  - Продукты на вывод
  - Продукты в дефектуре
  - Округление до коробов

- **Расчет отгрузок:**
  - По складам маркетплейсов
  - С учетом коэффициента покрытия
  - С применением ограничений

- **История прогнозов:**
  - Сохранение всех прогнозов
  - Сравнение моделей
  - Просмотр исторических данных

## Установка

```bash
pip install -r requirements_forecasting.txt
```

## Структура данных

### Входные данные (в папке `data/`):

1. **wb_sales.csv** - продажи на Wildberries
   - Колонки: Годы (Дата), Месяцы (Дата), Дата, solo-code, SKU, Количество упак., Унифицированный solo-code

2. **ozon_sales.csv** - продажи на Ozon
   - Колонки: Годы (Дата), Месяцы (Дата), Дата, solo-code, SKU, Количество упак., Унифицированный solo-code

3. **wb_stocks.csv** - остатки на Wildberries
   - Колонки: Годы (Дата), Месяцы (Дата), Дата, Склад, solo-code, SKU, Остаток, Унифицированный solo-code

4. **ozon_stocks.csv** - остатки на Ozon
   - Колонки: Годы (Дата), Месяцы (Дата), Дата, Склад, SKU, solo-code, Остаток, Унифицированный solo-code

5. **our_stocks.csv** - остатки на нашем складе
   - Колонки: Годы (Дата), Месяцы (Дата), Дата, Унифицированный solo-code, SKU, Остаток

6. **withdraw.csv** - список продуктов на вывод
   - Колонки: Унифицированный solo-code, SKU

7. **defecture.csv** - список продуктов в дефектуре
   - Колонки: Унифицированный solo-code, SKU, Дата окончания дефектуры

## Использование

### Базовое использование:

```python
from forecasting.forecaster import SalesForecaster

# Создание прогнозировщика
forecaster = SalesForecaster(data_path='data', forecast_months=18)

# Загрузка данных
forecaster.load_data()

# Подготовка моделей
forecaster.prepare_models()

# Запуск прогнозирования
results = forecaster.run_full_forecast(marketplace='wb')

# Результаты
best_forecast = results['best_forecast']
shipments = results['shipments']
evaluation = results['evaluation_summary']
```

### Из командной строки:

```bash
python -m forecasting.main --data-path data --marketplace both --months 18
```

### Параметры:

- `--data-path`: Путь к папке с данными (по умолчанию: `data`)
- `--marketplace`: Маркетплейс (`wb`, `ozon`, `both`)
- `--months`: Количество месяцев для прогноза (по умолчанию: 18)
- `--product`: Унифицированный код продукта (если не указан - для всех)

## Настройка

### Коэффициент покрытия для отгрузок:

```python
forecaster.shipment_calculator.set_coverage_coefficient(1.5)
```

### Размеры коробов:

```python
box_sizes = {
    'product_code_1': 24,
    'product_code_2': 56,
    'product_code_3': 72
}
forecaster.constraints.set_box_sizes(box_sizes)
```

### Даты черной пятницы:

```python
ozon_bf = ['2023-11-24', '2024-11-29']
wb_bf = ['2023-11-24', '2024-11-29']
forecaster.calendar_features.add_black_friday_dates(ozon_bf, wb_bf)
```

## Просмотр истории прогнозов

```python
from forecasting.forecast_manager import ForecastManager

manager = ForecastManager()

# Получить историю прогнозов
history = manager.get_forecast_history(marketplace='wb', model_name='prophet')

# Сравнить прогнозы разных моделей
comparison = manager.compare_forecasts('wb', 'unified_code_123', actual_sales)
```

## Структура проекта

```
forecasting/
├── __init__.py
├── data_loader.py          # Загрузка данных
├── calendar_features.py    # Календарные признаки
├── forecaster.py           # Главный модуль
├── model_evaluator.py      # Оценка моделей
├── shipment_calculator.py  # Расчет отгрузок
├── constraints.py          # Ограничения
├── forecast_manager.py     # Управление прогнозами
├── main.py                 # CLI интерфейс
└── models/
    ├── __init__.py
    ├── baseline.py         # Baseline модели
    ├── linear_regression.py # Линейная регрессия
    ├── arima.py            # ARIMA/SARIMA/SARIMAX
    └── prophet.py          # Prophet
```

## Примечания

- Система использует **Унифицированный solo-code** для идентификации продуктов
- Прогноз создается на 18 месяцев вперед (rolling forecast)
- Для каждого продукта автоматически выбирается лучшая модель на основе MAPE
- Отрицательные прогнозы автоматически обнуляются
- Отгрузки округляются до размера короба


