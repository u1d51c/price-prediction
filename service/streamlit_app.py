"""Streamlit UI поверх FastAPI-сервиса.

Пользователь выбирает актив, нажимает кнопку — UI дёргает FastAPI-эндпоинт
/predict/{asset} (который грузит PRD-модель из MLflow) и показывает прогноз
направления с вероятностями.

Запуск (FastAPI должен быть поднят на :8766):
    streamlit run service/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8766")

ASSETS = {
    "Brent (нефть BZ=F)": "brent",
    "WTI (нефть CL=F)": "wti",
    "Bitcoin (BTC-USD)": "btc",
}
DIR_RU = {"up": "📈 Рост", "down": "📉 Падение", "flat": "➡️ Без движения"}
DIR_COLOR = {"up": "#16a34a", "down": "#dc2626", "flat": "#6b7280"}

st.set_page_config(page_title="Прогноз направления цены", page_icon="📊", layout="centered")

st.title("📊 Прогноз направления движения цены")
st.caption("Brent / WTI / Bitcoin — классификация up / down / flat на следующий торговый день. "
           "Модель: RandomForest + ценовые + GDELT + FinBERT-признаки, загружается из MLflow (тег PRD).")

col1, col2 = st.columns([2, 1])
with col1:
    asset_label = st.selectbox("Актив", list(ASSETS.keys()))
with col2:
    n_days = st.number_input("Дней истории", min_value=1, max_value=14, value=5)

asset = ASSETS[asset_label]

if st.button("Получить прогноз", type="primary", use_container_width=True):
    with st.spinner("Считаю фичи и запрашиваю PRD-модель из MLflow..."):
        try:
            r = httpx.get(f"{API_URL}/predict/{asset}", params={"n_days": int(n_days)}, timeout=180)
        except Exception as exc:
            st.error(f"Сервис недоступен: {exc}")
            st.stop()

    if r.status_code != 200:
        st.error(f"Ошибка {r.status_code}: {r.text}")
        st.stop()

    data = r.json()
    preds = data["predictions"]
    last = preds[-1]

    # крупный вывод последнего прогноза
    st.markdown("### Прогноз на ближайший торговый день")
    direction = last["prediction"]
    st.markdown(
        f"<div style='font-size:2rem;font-weight:700;color:{DIR_COLOR[direction]}'>"
        f"{DIR_RU[direction]}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"На основе данных по {last['date']}")

    # вероятности классов
    probs = last["probabilities"]
    st.markdown("#### Вероятности классов")
    prob_df = pd.DataFrame({
        "класс": [DIR_RU[k] for k in probs],
        "вероятность": list(probs.values()),
    }).set_index("класс")
    st.bar_chart(prob_df, height=200)

    # таблица истории прогнозов
    st.markdown("#### Прогнозы за выбранный период")
    hist = pd.DataFrame([
        {"дата": p["date"], "прогноз": DIR_RU[p["prediction"]],
         **{f"P({k})": round(v, 3) for k, v in p["probabilities"].items()}}
        for p in preds
    ])
    st.dataframe(hist, use_container_width=True, hide_index=True)

    # метаданные модели
    with st.expander("Детали модели"):
        st.write({
            "актив": data["asset"],
            "версия PRD (MLflow)": data["model_version"],
            "число признаков": data["n_features"],
            "train/test split": data["split_date"],
            "время обработки, мс": round(data.get("processing_ms", 0), 1),
        })

st.divider()
st.caption("⚠️ Прогноз носит исследовательский характер. На дневном горизонте рынок "
           "близок к эффективному: реальный edge модели ~+2% над случайным выбором. "
           "Не является инвестиционной рекомендацией.")
