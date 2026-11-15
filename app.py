
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="CurrencyMind AI",
    page_icon="💱",
    layout="wide"
)

st.title("💱 CurrencyMind AI")
st.caption("Учебная система анализа валютных курсов")

CURRENCIES = {
    "USD": "Доллар США",
    "EUR": "Евро",
    "GBP": "Британский фунт",
    "PLN": "Польский злотый",
    "CHF": "Швейцарский франк"
}

@st.cache_data(ttl=3600)
def get_currency_data(code):
    start = "20200101"
    end = date.today().strftime("%Y%m%d")

    url = (
        "https://bank.gov.ua/NBU_Exchange/"
        "exchange_site"
    )

    params = {
        "start": start,
        "end": end,
        "valcode": code.lower(),
        "sort": "exchangedate",
        "order": "asc",
        "json": ""
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError("Нет исторических данных")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(
        df["exchangedate"],
        dayfirst=True
    )

    df["rate"] = pd.to_numeric(df["rate"])

    df = df[["date", "rate"]]
    df = df.drop_duplicates("date")
    df = df.sort_values("date")
    df = df.set_index("date")

    return df


def create_features(df):
    data = df.copy()

    # Предыдущие значения курса
    for lag in [1, 2, 3, 5, 10]:
        data[f"lag_{lag}"] = data["rate"].shift(lag)

    # Изменения курса
    data["change_1"] = data["rate"].pct_change(1)
    data["change_5"] = data["rate"].pct_change(5)

    # Скользящие средние
    data["mean_5"] = data["rate"].rolling(5).mean()
    data["mean_10"] = data["rate"].rolling(10).mean()

    # Волатильность
    data["volatility_5"] = (
        data["change_1"].rolling(5).std()
    )

    # Следующее опубликованное значение курса
    data["target"] = data["rate"].shift(-1)

    return data.dropna()


FEATURES = [
    "rate",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "change_1",
    "change_5",
    "mean_5",
    "mean_10",
    "volatility_5"
]


def analyze_currency(df):
    data = create_features(df)

    if len(data) < 100:
        raise ValueError(
            "Недостаточно данных для анализа"
        )

    X = data[FEATURES]
    y = data["target"]

    # Хронологическое разделение:
    # старые данные — обучение,
    # более новые — проверка.
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    models = {
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        error = mean_absolute_error(
            y_test,
            predictions
        )

        results[name] = {
            "model": model,
            "mae": error
        }

    # Проверка простого прогноза:
    # следующее значение = текущее значение.
    baseline = X_test["rate"].values

    baseline_error = mean_absolute_error(
        y_test,
        baseline
    )

    # Выбираем лучшую модель только среди
    # моделей с меньшей ошибкой, чем baseline.
    candidates = {
        name: result
        for name, result in results.items()
        if result["mae"] < baseline_error
    }

    if candidates:
        best_name = min(
            candidates,
            key=lambda name: candidates[name]["mae"]
        )
        best_model = candidates[best_name]["model"]
        selected_name = best_name
    else:
        best_model = None
        selected_name = "Baseline"

    # Для прогноза используем последнюю
    # доступную строку признаков.
    latest_features = (
    build_feature_rows(df)
    .dropna(subset=FEATURES)[FEATURES]
    .iloc[[-1]]
    )

    if best_model is not None:
        forecast = float(
            best_model.predict(latest_features)[0]
        )
    else:
        forecast = float(df["rate"].iloc[-1])

    return {
        "forecast": forecast,
        "selected": selected_name,
        "baseline_mae": baseline_error,
        "results": results,
        "latest": float(df["rate"].iloc[-1]),
        "test_dates": y_test.index,
        "test_actual": y_test,
    }


currency = st.selectbox(
    "Выбери валюту",
    list(CURRENCIES.keys())
)

st.write(
    f"Анализируем: {CURRENCIES[currency]} / UAH"
)

try:
    with st.spinner("Получаем данные и обучаем модели..."):
        df = get_currency_data(currency)
        result = analyze_currency(df)

    last_rate = result["latest"]
    forecast = result["forecast"]

    change = forecast - last_rate
    percent = change / last_rate * 100

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Последний официальный курс",
        f"{last_rate:.4f} UAH"
    )

    c2.metric(
        "Учебный прогноз",
        f"{forecast:.4f} UAH",
        f"{percent:+.3f}%"
    )

    c3.metric(
        "Выбранная модель",
        result["selected"]
    )

    st.subheader("История курса")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["rate"],
        mode="lines",
        name="Исторический курс"
    ))

    fig.update_layout(
        xaxis_title="Дата",
        yaxis_title="UAH",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Сравнение моделей")

    comparison = []

    comparison.append({
        "Модель": "Baseline",
        "MAE": result["baseline_mae"]
    })

    for name, item in result["results"].items():
        comparison.append({
            "Модель": name,
            "MAE": item["mae"]
        })

    st.dataframe(
        pd.DataFrame(comparison),
        use_container_width=True
    )

    st.info(
        "Прогноз относится к следующему "
        "опубликованному значению официального курса. "
        "Это экспериментальная оценка, а не гарантия."
    )

except Exception as error:
    st.error(f"Не удалось выполнить анализ: {error}")

st.divider()

st.caption(
    "Источник валютных данных: Национальный банк Украины. "
    "Проект предназначен для обучения и исследования."
  )


def build_feature_rows(df):
    data = df.copy()

    for lag in [1, 2, 3, 5, 10]:
        data[f"lag_{lag}"] = data["rate"].shift(lag)

    data["change_1"] = data["rate"].pct_change(1)
    data["change_5"] = data["rate"].pct_change(5)

    data["mean_5"] = data["rate"].rolling(5).mean()
    data["mean_10"] = data["rate"].rolling(10).mean()

    data["volatility_5"] = (
        data["change_1"].rolling(5).std()
    )

    return data


def create_features(df):
    data = build_feature_rows(df)
    data["target"] = data["rate"].shift(-1)

    return data.dropna(subset=FEATURES + ["target"])
  
