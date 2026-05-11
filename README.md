# Предсказание направления движения цен на сырьевые активы и BTC по новостям и социальным медиа

**Годовой проект ВШЭ (трек ML/DL).**
Тема объединяет №17 *"Предсказание стоимости акций"* и №13 *"Предсказание движения цен на фьючерсы на основе текстовых данных"*. Активы: фьючерсы на **нефть Brent (BZ=F)** и **WTI (CL=F)**, а также **Bitcoin (BTC-USD)** как цифровой коммодити.

## Постановка задачи

**Целевая переменная:** дневное направление движения цены - `up / down / flat` (3-классовая классификация).
**Признаки:**

1. **Числовые** - OHLCV, доходности, реализованная волатильность, технические индикаторы.
2. **Текстовые** - финансовые новости (GDELT 2.0, NewsAPI, Alpha Vantage) и обсуждения в Reddit (`r/Bitcoin`, `r/CryptoCurrency`, `r/commodities`, `r/energy`, `r/investing`).
3. **Sentiment-фичи** - на старте VADER (быстро, без GPU), далее FinBERT и доменно-настроенный CrudeBERT для нефти.

**Метрика-якорь:** macro-F1 (классы несбалансированы - flat редок), вспомогательные - accuracy, balanced accuracy, ROC-AUC по парам классов, profit/Sharpe ratio в backtest.

## Почему именно так

Выбор обоснован свежей литературой (полный список ниже):

* Подтверждено, что **новости и социальные медиа улучшают direction-классификацию** в нефти и BTC (Hashamia & Maldonado, 2025; Gurgul, Lessmann, Härdle, 2023; Elshendy et al., 2021).
* **Доменная адаптация** sentiment-моделей под нефть даёт ощутимый прирост поверх FinBERT (Kaplan et al., 2023, CrudeBERT).
* **Гибрид "цены + текст"** с attention/LSTM-стримами показывает большой gap к baseline-ам - например, AUC 0.94 против 0.34-0.57 у логистики/RF (Ghali et al., 2025, [arXiv:2508.06497]).

## Структура репозитория

```
.
├── configs/               # YAML-конфиги (данные, обучение, эксперименты)
├── data/
│   ├── raw/               # сырые скачанные данные (не коммитим)
│   ├── processed/         # очищенные/сопоставленные по дате (не коммитим)
│   └── external/          # справочники, готовые датасеты с Kaggle
├── notebooks/
│   ├── EDA.ipynb              # чекпоинт 2 - разведочный анализ
│   ├── ML.ipynb               # чекпоинт 3 - baseline ML
│   ├── ML_Experiments.ipynb   # чекпоинт 5 - улучшения ML
│   └── DL_Experiments.ipynb   # чекпоинт 6 - нейросети
├── src/
│   ├── data_collection/   # сбор цен и текстов
│   ├── features/          # инженерия признаков, sentiment
│   ├── models/            # baselines, гибридные модели
│   └── utils/             # вспомогательное
├── reports/
│   ├── figures/           # графики из notebooks
│   └── literature/        # PDF/заметки по статьям
├── tests/                 # юнит-тесты ключевых модулей
├── requirements.txt
└── README.md
```

## Воспроизведение

```bash
# 1. Создать виртуальное окружение и установить зависимости
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Прописать API-ключи (опционально, для NewsAPI/Reddit)
cp .env.example .env  # затем отредактировать

# 3. Скачать данные
python -m src.data_collection.fetch_prices
python -m src.data_collection.fetch_gdelt     # без ключей
python -m src.data_collection.fetch_reddit    # требует .env

# 4. Открыть EDA
jupyter lab notebooks/EDA.ipynb
```

## Литература

Все ключевые методические решения проекта опираются на следующие работы:

1. **Kaplan, H., Mundani, R.-P., Rölke, H., Weichselbraun, A. (2023).** *CrudeBERT: Applying Economic Theory towards fine-tuning Transformer-based Sentiment Analysis Models to the Crude Oil Market.* Proceedings of the 25th ICEIS, pp. 324-334. arXiv: [2305.06140](https://arxiv.org/abs/2305.06140).
   *- основа sentiment-блока для нефти; доменное дообучение FinBERT по supply/demand-разметке заголовков.*

2. **Hashamia, R., Maldonado, F. (2025).** *Can News Predict the Direction of Oil Price Volatility? A Language Model Approach with SHAP Explanations.* arXiv: [2508.20707](https://arxiv.org/abs/2508.20707).
   *- постановка direction-classification по новостям, ансамбль FinBERT/FastText/LLaMA/Gemini, McNemar-тесты значимости, SHAP-интерпретация.*

3. **Elshendy, M., Fronzetti Colladon, A., Battistoni, E., Gloor, P. A. (2021).** *Using four different online media sources to forecast the crude oil price.* arXiv: [2105.09154](https://arxiv.org/abs/2105.09154).
   *- GDELT (число статей, тон), Twitter (сложность языка), Wikipedia (просмотры), Google Trends; ARIMAX. Подтверждает добавочную ценность GDELT-сигналов.*

4. **Gurgul, V., Lessmann, S., Härdle, W. K. (2023/2024).** *Deep Learning and NLP in Cryptocurrency Forecasting: Integrating Financial, Blockchain, and Social Media Data.* arXiv: [2311.14759](https://arxiv.org/abs/2311.14759). Опубликовано в *International Journal of Forecasting* (2025).
   *- BTC/ETH, BART-MNLI zero-shot bullish/bearish, прирост по Sharpe ratio от текстовых фичей.*

5. **Ghali, M., Pang, R., Molina, A., Gershenson-Garcia, F., Won, J. (2025).** *Forecasting Commodity Price Shocks Using Temporal and Semantic Fusion of Prices Signals and Agentic Generative AI Extracted Economic News.* arXiv: [2508.06497](https://arxiv.org/abs/2508.06497).
   *- dual-stream LSTM + attention, fusion цен и текстовых эмбеддингов; AUC 0.94 vs 0.34-0.57 у baseline. Базовая архитектура для чекпоинта 6.*

6. **Wei, Y., Wang, J. и др. (2025).** *Crude oil price fluctuation forecasting incorporating news sentiment based on improved sentiment lexicon.* *J. of King Saud University CIS*, [s44443-025-00289-8](https://link.springer.com/article/10.1007/s44443-025-00289-8).
   *- словарный sentiment-подход как сравнение с трансформерными.*

7. **Beyond Polarity (2025).** *Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction.* arXiv: [2603.11408](https://arxiv.org/html/2603.11408).
   *- пять размерностей sentiment (релевантность, полярность, интенсивность, неопределённость, обращённость к будущему).*

> Цитаты с указанием страниц или разделов будут проставлены непосредственно в коде/выводах каждого ноутбука, где соответствующее методическое решение применяется.

## Команда

* Студент: Демьянова Юлия
* Куратор: Гринберг Петр

## Лицензия

Код учебного проекта. Использование сторонних данных регулируется условиями их провайдеров (Yahoo Finance, GDELT, Reddit, NewsAPI).
