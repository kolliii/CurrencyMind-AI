
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
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

    df["rate"] = pd.to_numeric(
    df["rate"],
    errors="coerce"
)

    df = df[["date", "rate"]]
    df = df.drop_duplicates("date")
    df = df.sort_values("date")
    df = df.set_index("date")

    # Перевірка даних
    if df.empty:
        raise ValueError("Дані від НБУ відсутні")

    if df["rate"].isna().any():
        raise ValueError("У даних є пропущені значення")

    if (df["rate"] <= 0).any():
        raise ValueError("Виявлено некоректні значення курсу")

    return df
    



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

        # Значення, яке потрібно оцінити на наступному спостереженні
        data["target"] = data["rate"].shift(-1)

        # Дата цього наступного спостереження
        data["target_date"] = pd.Series(
        data.index,
        index=data.index
        ).shift(-1)

    return data.dropna(
        subset=FEATURES + ["target", "target_date"]
    )
    


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

    if len(data) < 200:
        raise ValueError(
            "Недостатньо даних для Walk-Forward Validation. "
            "Потрібно щонайменше 200 записів."
        )

    X = data[FEATURES]
    y = data["target"]

    # Налаштування Walk-Forward Validation
    splitter = TimeSeriesSplit(
        n_splits=5,
        test_size=30
    )

    # Функція створення моделей
    def make_models():
        return {
            "Ridge": make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0)
            ),

            "Random Forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42
            )
        }

    # Зберігання помилок моделей
    model_errors = {
        "Ridge": [],
        "Random Forest": []
    }
    # Збереження фактичних і прогнозованих значень
    prediction_records = {
    "Baseline": [],
    "Ridge": [],
    "Random Forest": []
    }

    baseline_errors = []
    
    fold_results = []

    # Послідовна перевірка
    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(X),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Baseline: прогноз наступного значення
        baseline_predictions = X_test["rate"].values

        baseline_error = mean_absolute_error(
            y_test,
            baseline_predictions
        )

        baseline_errors.append(baseline_error)
        # Збереження результатів Baseline
        for i in range(len(test_idx)):
            prediction_records["Baseline"].append({
            "date": data["target_date"].iloc[test_idx[i]],
            "actual": float(y_test.iloc[i]),
            "predicted": float(baseline_predictions[i]),
            "fold": fold_number
        })

        fold_record = {
            "Перевірка": fold_number,
            "Baseline MAE": baseline_error
        }

        # Навчання та перевірка моделей
        for name, model in make_models().items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            for i in range(len(test_idx)):
            prediction_records[name].append({
               "date": data["target_date"].iloc[test_idx[i]],
               "actual": float(y_test.iloc[i]),
               "predicted": float(predictions[i]),
               "fold": fold_number
            })

            error = mean_absolute_error(
                y_test,
                predictions
            )

            model_errors[name].append(error)

            fold_record[name] = error

        fold_results.append(fold_record)

    # Обчислення середньої помилки
    results = {}

    for name, errors in model_errors.items():

        results[name] = {
            "mae": float(np.mean(errors)),
            "fold_mae": errors
        }

    baseline_error = float(
        np.mean(baseline_errors)
    )

    # Вибір моделі за середньою помилкою
    best_name = min(
        results,
        key=lambda name: results[name]["mae"]
    )

    # Фінальне навчання на всій доступній історії
    best_model = make_models()[best_name]

    best_model.fit(X, y)

    # Останні доступні ознаки
    latest_features = (
        build_feature_rows(df)
        .dropna(subset=FEATURES)[FEATURES]
        .iloc[[-1]]
    )

    # Навчальна оцінка наступного значення
    forecast = float(
        best_model.predict(latest_features)[0]
    )

    latest = float(df["rate"].iloc[-1])

    return {
        "forecast": forecast,
        "selected": best_name,
        "selected_mae": results[best_name]["mae"],
        "baseline_mae": baseline_error,
        "baseline_better": (
            baseline_error <= results[best_name]["mae"]
        ),
        "results": results,
        "latest": latest,
        "fold_results": fold_results,
        "prediction_records": prediction_records
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

    # Основные показатели
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

    # Проверка качества моделей
    st.subheader("Качество моделей")

    st.write(
        f"MAE выбранной модели: "
        f"{result['selected_mae']:.6f}"
    )

    st.write(
        f"MAE Baseline: "
        f"{result['baseline_mae']:.6f}"
    )

    if result["baseline_better"]:
        st.warning(
            "Baseline показал меньшую ошибку. "
            "Результат ML-модели следует трактовать осторожно."
        )
    else:
        st.success(
            "ML-модель показала меньшую ошибку, "
            "чем Baseline на проверочных данных."
        )

    # График курса
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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Сравнение моделей
    st.subheader("Сравнение моделей")

    comparison = [{
        "Модель": "Baseline",
        "MAE": result["baseline_mae"]
    }]

    for name, item in result["results"].items():
        comparison.append({
            "Модель": name,
            "MAE": item["mae"]
        })

    st.dataframe(
        pd.DataFrame(comparison),
        use_container_width=True
    )
    
    # Графік фактичних і прогнозованих значень
    st.subheader("Фактичні значення та прогнози моделей")

        actual_df = pd.DataFrame(
        result["prediction_records"]["Baseline"]
        ).sort_values("date")

        fig_predictions = go.Figure()

    # Фактичний курс
        fig_predictions.add_trace(go.Scatter(
        x=actual_df["date"],
        y=actual_df["actual"],
        mode="lines",
        name="Фактичне значення"
    ))

    # Прогнози кожної моделі
    for model_name in ["Baseline", "Ridge", "Random Forest"]:

        model_df = pd.DataFrame(
        result["prediction_records"][model_name]
        ).sort_values("date")

        fig_predictions.add_trace(go.Scatter(
        x=model_df["date"],
        y=model_df["predicted"],
        mode="lines",
        name=f"Прогноз {model_name}"
   ))

    fig_predictions.update_layout(
    xaxis_title="Дата",
    yaxis_title="Офіційний курс, UAH",
    template="plotly_dark",
    hovermode="x unified"
  )

  st.plotly_chart(
     fig_predictions,
     use_container_width=True
  )

  
# Графік помилок на кожній перевірці
    st.subheader("Помилки моделей за перевірками")

fold_chart = go.Figure()

for column in [
    "Baseline MAE",
    "Ridge",
    "Random Forest"
]:
    fold_chart.add_trace(go.Bar(
        x=fold_df["Перевірка"],
        y=fold_df[column],
        name=column
    ))

fold_chart.update_layout(
    xaxis_title="Номер перевірки",
    yaxis_title="MAE, UAH",
    barmode="group",
    template="plotly_dark"
)

st.plotly_chart(
    fold_chart,
    use_container_width=True
)

    st.subheader("Результати Walk-Forward Validation")

    fold_df = pd.DataFrame(result["fold_results"])

    st.dataframe(
    fold_df,
    use_container_width=True
    )


    st.info(
        "Прогноз относится к следующему опубликованному "
        "значению официального курса. "
        "Это экспериментальная оценка, а не гарантия."
    )

except Exception as error:
    st.error(
        f"Не удалось выполнить анализ: {error}"
    )

st.divider()

st.caption(
    "Источник валютных данных: Национальный банк Украины. "
    "Проект предназначен для обучения и исследования."
    )



    
