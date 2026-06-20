import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Настройки страницы Streamlit
st.set_page_config(page_title = "Прогноз общего спроса на жильё ", layout = "wide")


#загрузка данных и модели
@st.cache_resource
def load_model_and_data():
    df_train = pd.read_csv('df_full_train.csv')
    df_eval = pd.read_csv('df_eval_clean.csv')

    df_train['region_code'] = df_train['region_code'].astype(str).str.strip()
    df_eval['region_code'] = df_eval['region_code'].astype(str).str.strip()

    model = CatBoostRegressor()
    model.load_model('cb_model.cbm')
    return df_train, df_eval, model


df_full_train, df_eval_clean, final_model = load_model_and_data()
TARGET_COL = 'total_demand_mil_rub'

# боковое меню
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел приложения:",
    ["Прогноз по регионам", "EDA по датасету", "Метрики качества моделей"]
)

# ============================================
# Прогноз общего объема рынка по регионам в денежном выражении

if page == "Прогноз по регионам":
    st.title("Прогноз общего объема рынка по регионам в денежном выражении")
    st.write("Помесячный анализ и прогноз совокупного оборота рынка недвижимости (млн руб.).")

    st.sidebar.markdown("---")
    st.sidebar.header("Параметры региона")

    # Сортировка кодов регионов для красивого списка в интерфейсе
    all_regions = sorted(df_eval_clean['region_code'].unique(),
                         key=lambda x: int(float(str(x).replace(',', '.'))) if str(x).replace('.',
                                                                                              '').isdigit() else str(x))
    selected_region = st.sidebar.selectbox("Выберите регион:", all_regions)

    df_region_metrics = df_eval_clean[df_eval_clean['region_code'] == selected_region].copy()

    # Динамический расчет локальных метрик в сайдбар
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Точность для региона {selected_region}:")
    if not df_region_metrics.empty:
        df_3m = df_region_metrics[df_region_metrics['month'] <= 3]
        if not df_3m.empty:
            st.sidebar.success(
                f"**Горизонт 3 месяца:**\n* MAPE: {round(mean_absolute_percentage_error(df_3m[TARGET_COL], df_3m['pred_total_demand']) * 100, 2)}%\n* RMSE: {round(np.sqrt(mean_squared_error(df_3m[TARGET_COL], df_3m['pred_total_demand'])), 2)}")

        df_6m = df_region_metrics[df_region_metrics['month'] <= 6]
        if not df_6m.empty:
            st.sidebar.warning(
                f"**Горизонт 6 месяцев:**\n* MAPE: {round(mean_absolute_percentage_error(df_6m[TARGET_COL], df_6m['pred_total_demand']) * 100, 2)}%\n* RMSE: {round(np.sqrt(mean_squared_error(df_6m[TARGET_COL], df_6m['pred_total_demand'])), 2)}")

    # Фильтруем прогнозные данные по выбранному региону на 6 месяцев
    region_eval_data = df_eval_clean[
        (df_eval_clean['region_code'] == selected_region) & (df_eval_clean['month'] <= 6)].copy()

    if not region_eval_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Последний факт объема (Июнь 2025)",
                      f"{round(region_eval_data[TARGET_COL].iloc[-1], 2)} млн руб.")
        with col2:
            st.metric("Прогноз модели (Июнь 2025)",
                      f"{round(region_eval_data['pred_total_demand'].iloc[-1], 2)} млн руб.")

        # Подгружаем историю 2024 года для выбранного региона
        history_df = df_full_train[df_full_train['region_code'] == selected_region].copy()
        history_monthly = history_df[history_df['year'] == 2024].groupby(['year', 'month'])[
            TARGET_COL].sum().reset_index()
        history_monthly['date'] = pd.to_datetime(
            history_monthly['year'].astype(str) + '-' + history_monthly['month'].astype(str).str.zfill(2) + '-01')

        # Готовим временные оси для 2025 года
        region_eval = region_eval_data.copy()
        region_eval['date'] = pd.to_datetime(
            region_eval['year'].astype(str) + '-' + region_eval['month'].astype(str).str.zfill(2) + '-01')

        last_2024_point = history_monthly.sort_values('date').tail(1)
        fact_line = pd.concat([pd.DataFrame({'date': last_2024_point['date'], TARGET_COL: last_2024_point[TARGET_COL]}),
                               region_eval[['date', TARGET_COL]]]).sort_values('date')
        pred_line = pd.concat(
            [pd.DataFrame({'date': last_2024_point['date'], 'pred_total_demand': last_2024_point[TARGET_COL]}),
             region_eval[['date', 'pred_total_demand']]]).sort_values('date')


        # 95 интервал

        Q_ERROR_CONFORMAL = 27322.37

        # Строим коридоры доверительного интервала вокруг прогноза
        pred_line['lower_bound'] = pred_line['pred_total_demand'] - Q_ERROR_CONFORMAL
        pred_line['upper_bound'] = pred_line['pred_total_demand'] + Q_ERROR_CONFORMAL

        # Оборот рынка в деньгах не может упасть ниже нуля
        pred_line['lower_bound'] = pred_line['lower_bound'].clip(lower=0)
        # -----------------------------------------------------------------

        # Отрисовка графика
        fig, ax = plt.subplots(figsize=(14, 5.5))

        # Основные тренды
        ax.plot(history_monthly['date'], history_monthly[TARGET_COL], label='История 2024 (Факт)', color='#7f8c8d',
                marker='o', alpha=0.5)
        ax.plot(fact_line['date'], fact_line[TARGET_COL], label='Реальный оборот 2025 (Факт)', color='red',
                marker='o', linewidth=2)
        ax.plot(pred_line['date'], pred_line['pred_total_demand'], label='Прогноз CatBoost_log',
                color='blue', linestyle='--', marker='o', linewidth=2)

        # закраска 95% интервала для прогноза
        ax.fill_between(
            pred_line['date'],
            pred_line['lower_bound'],
            pred_line['upper_bound'],
            color='blue',
            alpha=0.10,
            label='95% Доверительный интервал'
        )

        # Вертикальные маркеры горизонтов планирования
        ax.axvline(x=pd.to_datetime('2025-03-01'), color='green', linestyle=':', alpha=0.9, label='Горизонт 3 месяца')
        ax.axvline(x=pd.to_datetime('2025-06-01'), color='orange', linestyle=':', alpha=0.9, label='Горизонт 6 месяцев')

        ax.set_ylabel('Млн рублей')
        ax.set_title(f'Прогноз совокупного денежного объема рынка недвижимости для региона {selected_region}',
                     fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.15)

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(f"Нет доступных данных 2025 года для построения прогноза по региону {selected_region}.")

# экран eda
elif page == "EDA по датасету":
    st.title("Разведочный анализ данных (EDA)")
    st.write("Статистический обзор признаков исторической выборки до 2024 года включительно.")

    #  Сводные метрики датасета
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего исторических записей", len(df_full_train))
    col2.metric("Количество уникальных регионов", df_full_train['region_code'].nunique())
    col3.metric("Средний чек спроса по РФ", f"{round(df_full_train[TARGET_COL].mean(), 2)} млн руб.")

    st.markdown("---")

    #  График распределения таргета (с логарифмом и без)
    st.subheader("Распределение целевой переменной")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    sns.histplot(df_full_train[TARGET_COL], bins=30, kde=True, color='teal', ax=ax1)
    ax1.set_title("Исходное распределение спроса")

    sns.histplot(np.log1p(df_full_train[TARGET_COL]), bins=30, kde=True, color='crimson', ax=ax2)
    ax2.set_title("Логарифмированное распределение")
    st.pyplot(fig)
    plt.close(fig)

    # Просмотр пропусков и типов данных
    if st.checkbox("Показать структуру и типы колонок"):
        st.dataframe(df_full_train.dtypes.astype(str).to_frame(name='Тип данных'))

#метрики качесвта моделей

elif page == "Метрики качества моделей":
    st.title("Сравнение конфигураций моделей")
    st.write("Сводная таблица метрик по результатам тестирования различных подходов и моделей:")

    # Формируем таблицу на основе твоих данных
    metrics_data = {
        "Модель": ["Linear", "Ridge", "Median", "CatBoost", "CatBoost_log (Лучшая)"],
        "R²": [0.7026, 0.9084, 0.8871, 0.9730, 0.9759],
        "MAE (млн.руб)": [20787.3310, 12726.1163, 15952.4185, 6451.0500, 5511.6200],
        "RMSE (млн.руб)": [46020.7295, 25541.4495, 28355.0517, 13864.3100, 13093.5000],
        "MAPE (%)": [108.8296, 75.5468, 40.3213, 20.2200, 11.8100]
    }

    df_metrics = pd.DataFrame(metrics_data)

    # Интерактивная таблица с результатами
    st.dataframe(
        df_metrics.style.highlight_max(subset=["R²"], color="green")
        .highlight_min(subset=["MAE (млн.руб)", "RMSE (млн.руб)", "MAPE (%)"], color="green"),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Важность признаков (Feature Importance)")

    # важность признаков из обученной финальной модели
    importance = final_model.get_feature_importance()
    feature_names = final_model.feature_names_

    df_importance = pd.DataFrame({'Признак': feature_names, 'Важность': importance})
    df_importance = df_importance.sort_values('Важность', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x='Важность', y='Признак', data=df_importance, palette='viridis', ax=ax)
    ax.set_title("Топ-10 ключевых факторов, влияющих на спрос недвижимости")
    st.pyplot(fig)
    plt.close(fig)