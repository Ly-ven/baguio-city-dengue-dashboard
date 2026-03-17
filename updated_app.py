import json
import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    layout="wide"
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for dengue prediction and visualization")

ARTIFACTS_DIR = Path("artifacts")

DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos"
]


def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def safe_load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data
def load_artifacts():
    monthly = safe_read_csv(ARTIFACTS_DIR / "monthly_modeling_dataset.csv")
    model_comparison = safe_read_csv(ARTIFACTS_DIR / "model_comparison.csv")
    feature_importance = safe_read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    feature_sensitivity = safe_read_csv(ARTIFACTS_DIR / "feature_sensitivity.csv")
    forecast = safe_read_csv(ARTIFACTS_DIR / "forecast_5yr.csv")

    barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "barangay_monthly.csv")
    top_barangay_monthly = safe_read_csv(ARTIFACTS_DIR / "top_barangay_monthly.csv")
    top3_barangays_yearly = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_yearly.csv")
    top3_barangays_overall = safe_read_csv(ARTIFACTS_DIR / "top3_barangays_overall.csv")

    test_predictions = safe_read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    confusion_matrix_detail = safe_read_csv(ARTIFACTS_DIR / "confusion_matrix_detail.csv")
    climate_case_correlation = safe_read_csv(ARTIFACTS_DIR / "climate_case_correlation.csv")
    month_profile = safe_read_csv(ARTIFACTS_DIR / "month_profile.csv")

    # New barangay forecast/risk files
    forecast_barangay_ranking = safe_read_csv(ARTIFACTS_DIR / "forecast_barangay_ranking.csv")
    forecast_top3_barangays = safe_read_csv(ARTIFACTS_DIR / "forecast_top3_barangays.csv")
    barangay_risk_profile = safe_read_csv(ARTIFACTS_DIR / "barangay_risk_profile.csv")

    meta = safe_read_json(ARTIFACTS_DIR / "meta.json")

    return (
        monthly,
        model_comparison,
        feature_importance,
        feature_sensitivity,
        forecast,
        barangay_monthly,
        top_barangay_monthly,
        top3_barangays_yearly,
        top3_barangays_overall,
        test_predictions,
        confusion_matrix_detail,
        climate_case_correlation,
        month_profile,
        forecast_barangay_ranking,
        forecast_top3_barangays,
        barangay_risk_profile,
        meta,
    )


(
    monthly,
    model_comparison,
    feature_importance,
    feature_sensitivity,
    forecast,
    barangay_monthly,
    top_barangay_monthly,
    top3_barangays_yearly,
    top3_barangays_overall,
    test_predictions,
    confusion_matrix_detail,
    climate_case_correlation,
    month_profile,
    forecast_barangay_ranking,
    forecast_top3_barangays,
    barangay_risk_profile,
    meta,
) = load_artifacts()

model = safe_load_model(ARTIFACTS_DIR / "best_model.joblib")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("About")
st.sidebar.write(
    "This dashboard displays historical dengue cases, model results, feature contributions, "
    "and forecast outputs from your Google Colab workflow."
)

if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_val = meta.get("outbreak_threshold_cases", "N/A")
    if isinstance(threshold_val, (int, float)):
        st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f}")
    else:
        st.sidebar.info(f"Outbreak Threshold: {threshold_val}")
else:
    st.sidebar.warning("Metadata not found.")

st.sidebar.subheader("Upload files manually (optional)")

uploaded_monthly = st.sidebar.file_uploader("Upload monthly_modeling_dataset.csv", type=["csv"])
uploaded_model_comparison = st.sidebar.file_uploader("Upload model_comparison.csv", type=["csv"])
uploaded_feature_importance = st.sidebar.file_uploader("Upload feature_importance.csv", type=["csv"])
uploaded_feature_sensitivity = st.sidebar.file_uploader("Upload feature_sensitivity.csv", type=["csv"])
uploaded_forecast = st.sidebar.file_uploader("Upload forecast_5yr.csv", type=["csv"])
uploaded_barangay_monthly = st.sidebar.file_uploader("Upload barangay_monthly.csv", type=["csv"])
uploaded_top_barangay_monthly = st.sidebar.file_uploader("Upload top_barangay_monthly.csv", type=["csv"])
uploaded_top3_yearly = st.sidebar.file_uploader("Upload top3_barangays_yearly.csv", type=["csv"])
uploaded_top3_overall = st.sidebar.file_uploader("Upload top3_barangays_overall.csv", type=["csv"])
uploaded_test_predictions = st.sidebar.file_uploader("Upload test_predictions.csv", type=["csv"])
uploaded_cm_detail = st.sidebar.file_uploader("Upload confusion_matrix_detail.csv", type=["csv"])
uploaded_climate_corr = st.sidebar.file_uploader("Upload climate_case_correlation.csv", type=["csv"])
uploaded_month_profile = st.sidebar.file_uploader("Upload month_profile.csv", type=["csv"])
uploaded_forecast_barangay_ranking = st.sidebar.file_uploader("Upload forecast_barangay_ranking.csv", type=["csv"])
uploaded_forecast_top3_barangays = st.sidebar.file_uploader("Upload forecast_top3_barangays.csv", type=["csv"])
uploaded_barangay_risk_profile = st.sidebar.file_uploader("Upload barangay_risk_profile.csv", type=["csv"])
uploaded_meta = st.sidebar.file_uploader("Upload meta.json", type=["json"])
uploaded_model = st.sidebar.file_uploader("Upload best_model.joblib", type=["joblib", "pkl"])

# =========================
# USE UPLOADED FILES IF PROVIDED
# =========================
if uploaded_monthly is not None:
    monthly = pd.read_csv(uploaded_monthly)

if uploaded_model_comparison is not None:
    model_comparison = pd.read_csv(uploaded_model_comparison)

if uploaded_feature_importance is not None:
    feature_importance = pd.read_csv(uploaded_feature_importance)

if uploaded_feature_sensitivity is not None:
    feature_sensitivity = pd.read_csv(uploaded_feature_sensitivity)

if uploaded_forecast is not None:
    forecast = pd.read_csv(uploaded_forecast)

if uploaded_barangay_monthly is not None:
    barangay_monthly = pd.read_csv(uploaded_barangay_monthly)

if uploaded_top_barangay_monthly is not None:
    top_barangay_monthly = pd.read_csv(uploaded_top_barangay_monthly)

if uploaded_top3_yearly is not None:
    top3_barangays_yearly = pd.read_csv(uploaded_top3_yearly)

if uploaded_top3_overall is not None:
    top3_barangays_overall = pd.read_csv(uploaded_top3_overall)

if uploaded_test_predictions is not None:
    test_predictions = pd.read_csv(uploaded_test_predictions)

if uploaded_cm_detail is not None:
    confusion_matrix_detail = pd.read_csv(uploaded_cm_detail)

if uploaded_climate_corr is not None:
    climate_case_correlation = pd.read_csv(uploaded_climate_corr)

if uploaded_month_profile is not None:
    month_profile = pd.read_csv(uploaded_month_profile)

if uploaded_forecast_barangay_ranking is not None:
    forecast_barangay_ranking = pd.read_csv(uploaded_forecast_barangay_ranking)

if uploaded_forecast_top3_barangays is not None:
    forecast_top3_barangays = pd.read_csv(uploaded_forecast_top3_barangays)

if uploaded_barangay_risk_profile is not None:
    barangay_risk_profile = pd.read_csv(uploaded_barangay_risk_profile)

if uploaded_meta is not None:
    meta = json.load(uploaded_meta)

if uploaded_model is not None:
    model = joblib.load(uploaded_model)

# =========================
# REQUIRED CHECK
# =========================
if monthly is None:
    st.error("monthly_modeling_dataset.csv is required.")
    st.stop()

# =========================
# DATE CONVERSION
# =========================
for df_name in [
    "monthly",
    "forecast",
    "top_barangay_monthly",
    "barangay_monthly",
    "test_predictions",
    "forecast_barangay_ranking",
    "forecast_top3_barangays",
]:
    df_obj = locals().get(df_name)
    if df_obj is not None and "Date" in df_obj.columns:
        df_obj["Date"] = pd.to_datetime(df_obj["Date"], errors="coerce")
        locals()[df_name] = df_obj

# =========================
# HELPERS
# =========================
def safe_metric_value(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def outbreak_label_from_binary(x):
    return "Outbreak" if int(x) == 1 else "Non-outbreak"


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Barangay Analytics",
    "Model Results",
    "Feature Transparency",
    "Forecast & Prediction"
])

# =========================
# TAB 1 — OVERVIEW
# =========================
with tab1:
    st.header("Historical Dengue Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Months", len(monthly))
    col2.metric(
        "Total Cases",
        int(monthly["CHSO_cases"].sum()) if "CHSO_cases" in monthly.columns else 0
    )
    col3.metric(
        "Average Monthly Cases",
        safe_metric_value(monthly["CHSO_cases"].mean()) if "CHSO_cases" in monthly.columns else "N/A"
    )

    st.subheader("What is the model predicting?")
    if meta:
        st.info(
            f"Problem Definition: {meta.get('problem_definition', 'Monthly outbreak classification')}  \n"
            f"Outbreak Definition: {meta.get('outbreak_definition', 'Not available')}"
        )
    else:
        st.info("The model predicts whether a month is outbreak or non-outbreak.")

    st.subheader("Monthly Dengue Cases")
    if {"Date", "CHSO_cases"}.issubset(monthly.columns):
        fig_line = px.line(
            monthly,
            x="Date",
            y="CHSO_cases",
            markers=True,
            title="Monthly Dengue Cases in Baguio City (CHSO)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Year-Month Heatmap of Dengue Cases")
    if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
        heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Year-Month Heatmap of Dengue Cases"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Rainfall vs Relative Humidity Sized by Dengue Cases")
    if {"rainfall", "relative_humidity", "CHSO_cases"}.issubset(monthly.columns):
        hover_cols = ["Date"]
        if "temp_mid" in monthly.columns:
            hover_cols.append("temp_mid")

        fig_bubble = px.scatter(
            monthly,
            x="rainfall",
            y="relative_humidity",
            size="CHSO_cases",
            color="CHSO_cases",
            hover_data=hover_cols,
            title="Rainfall vs Relative Humidity Sized by Dengue Cases"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.subheader("Climate-Case Correlation")
    if climate_case_correlation is not None and not climate_case_correlation.empty:
        st.dataframe(climate_case_correlation, use_container_width=True)

        if {"feature", "pearson_corr_with_CHSO_cases"}.issubset(climate_case_correlation.columns):
            fig_corr = px.bar(
                climate_case_correlation,
                x="feature",
                y="pearson_corr_with_CHSO_cases",
                title="Climate-Case Correlation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("climate_case_correlation.csv not found or empty.")

    st.subheader("Average Monthly Profile")
    if month_profile is not None and not month_profile.empty:
        st.dataframe(month_profile, use_container_width=True)

        if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
            fig_month_profile = px.bar(
                month_profile,
                x="MonthName",
                y="CHSO_cases",
                title="Average CHSO Cases by Month"
            )
            st.plotly_chart(fig_month_profile, use_container_width=True)
    else:
        st.warning("month_profile.csv not found or empty.")

# =========================
# TAB 2 — BARANGAY ANALYTICS
# =========================
with tab2:
    st.header("Barangay Analytics")

    st.subheader("Top barangay by month")
    if top_barangay_monthly is not None:
        st.dataframe(top_barangay_monthly, use_container_width=True)

    st.subheader("Top Barangays by Dengue Cases")
    ranking_choice = st.radio(
        "Choose ranking view",
        ["Top 3 per year", "Top 3 overall"],
        horizontal=True
    )

    if ranking_choice == "Top 3 per year" and top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
        fig_tree = px.treemap(
            top3_barangays_yearly,
            path=["Year", "Barangay"],
            values="Barangay_cases",
            color="Barangay_cases",
            title="Top 3 Barangays per Year Based on Dengue Cases"
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    elif ranking_choice == "Top 3 overall" and top3_barangays_overall is not None and not top3_barangays_overall.empty:
        fig_top3 = px.bar(
            top3_barangays_overall,
            x="Barangay",
            y="Barangay_cases",
            text="Barangay_cases",
            title="Top 3 Barangays Overall Based on Dengue Cases"
        )
        st.plotly_chart(fig_top3, use_container_width=True)

    st.subheader("Barangay monthly records")
    if barangay_monthly is not None:
        st.dataframe(barangay_monthly, use_container_width=True)

# =========================
# TAB 3 — MODEL RESULTS
# =========================
with tab3:
    st.header("Model Comparison")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        display_cols = ["model", "accuracy", "precision", "recall", "f1_score"]
        available_display_cols = [c for c in display_cols if c in model_comparison.columns]
        st.dataframe(model_comparison[available_display_cols], use_container_width=True)

        st.subheader("Model Comparison by Metric")
        results_long = model_comparison.melt(
            id_vars="model",
            value_vars=["accuracy", "precision", "recall", "f1_score"],
            var_name="Metric",
            value_name="Score"
        )

        fig_model = px.bar(
            results_long,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Model Comparison by Metric"
        )
        st.plotly_chart(fig_model, use_container_width=True)

    st.subheader("How to read the metrics")
    st.markdown(
        """
- **Accuracy**: overall percentage of correct predictions  
- **Precision**: when the model says outbreak, how often it is correct  
- **Recall / Sensitivity**: among real outbreak months, how many the model correctly catches  
- **F1 Score**: balance between precision and recall  

Higher values are better.  

A model can have high accuracy but still miss outbreak months.  
That is why **precision, recall, and F1 score** must also be checked.
"""
    )

    st.subheader("Confusion Matrix")
    if model_comparison is not None and "confusion_matrix" in model_comparison.columns:
        selected_model_for_cm = st.selectbox(
            "Select model to view confusion matrix",
            model_comparison["model"].tolist(),
            index=0
        )

        selected_row = model_comparison[model_comparison["model"] == selected_model_for_cm].iloc[0]
        cm_raw = selected_row["confusion_matrix"]

        if isinstance(cm_raw, str):
            cm = np.array(ast.literal_eval(cm_raw))
        else:
            cm = np.array(cm_raw)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0 (Non-outbreak)", "Actual 1 (Outbreak)"],
            columns=["Predicted 0 (Non-outbreak)", "Predicted 1 (Outbreak)"]
        )

        st.dataframe(cm_df, use_container_width=True)

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto",
            title=f"Confusion Matrix - {selected_model_for_cm}"
        )
        fig_cm.update_xaxes(title="Predicted")
        fig_cm.update_yaxes(title="Actual")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("confusion_matrix column not found in model_comparison.csv.")

    st.subheader("How many months were correctly predicted?")
    if test_predictions is not None and not test_predictions.empty:
        total_test = len(test_predictions)
        correct_test = int(test_predictions["is_correct"].sum()) if "is_correct" in test_predictions.columns else None

        c1, c2 = st.columns(2)
        c1.metric("Test Set Months", total_test)
        c2.metric("Correct Predictions", correct_test if correct_test is not None else "N/A")

        st.dataframe(test_predictions, use_container_width=True)

# =========================
# TAB 4 — FEATURE TRANSPARENCY
# =========================
with tab4:
    st.header("What contributed to the prediction?")

    st.subheader("Feature Importance")
    if feature_importance is not None and not feature_importance.empty:
        st.dataframe(feature_importance, use_container_width=True)

        fig_importance = px.bar(
            feature_importance.head(15),
            x="importance_mean",
            y="feature",
            orientation="h",
            title="Top Contributing Features"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    st.subheader("Sensitivity Analysis")
    if feature_sensitivity is not None and not feature_sensitivity.empty:
        st.dataframe(feature_sensitivity, use_container_width=True)

        fig_sens = px.bar(
            feature_sensitivity,
            x="feature",
            y="delta_probability",
            title="Effect of +10% Change in Climate Variable on Outbreak Probability"
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    st.subheader("How to interpret this")
    st.markdown(
        """
- **Feature importance** shows which variables the model relied on most.  
- **Lagged case variables** mean the model uses recent dengue history.  
- **Sensitivity analysis** shows what happens to outbreak probability if rainfall, humidity, or temperature is changed.  
- These do not prove biological causation by themselves, but they help explain the model's behavior.
"""
    )

# =========================
# TAB 5 — FORECAST & PREDICTION
# =========================
with tab5:
    st.header("Forecast")

    if forecast is not None and not forecast.empty:
        st.dataframe(forecast.head(30), use_container_width=True)

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(
                index="Year",
                columns="Month",
                values="predicted_outbreak_probability"
            )
            fig_forecast_heat = px.imshow(
                forecast_heat,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                title="Forecast Heatmap of Outbreak Probability"
            )
            st.plotly_chart(fig_forecast_heat, use_container_width=True)

    # Added forecast barangay ranking section
    st.subheader("Top 3 Likely Barangays for Forecast Months")
    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        st.dataframe(forecast_top3_barangays, use_container_width=True)

        month_options = forecast_top3_barangays["Date"].dropna().astype(str).unique().tolist()
        selected_month = st.selectbox("Select forecast month for barangay ranking", month_options)

        selected_barangay_forecast = forecast_top3_barangays[
            forecast_top3_barangays["Date"].astype(str) == selected_month
        ].copy()

        fig_barangay_forecast = px.bar(
            selected_barangay_forecast,
            x="Barangay",
            y="predicted_barangay_cases_proxy",
            color="Barangay",
            title=f"Top 3 Likely Barangays - {selected_month}"
        )
        st.plotly_chart(fig_barangay_forecast, use_container_width=True)
    else:
        st.warning("forecast_top3_barangays.csv not found or empty.")

    st.subheader("Live Prediction")
    st.write("Enter the values below to predict one month.")

    if model is None:
        st.warning("Model file not found. Live prediction is unavailable.")
    else:
        if meta and "feature_columns" in meta:
            feature_columns = meta["feature_columns"]
        else:
            feature_columns = DEFAULT_FEATURE_COLS

        input_values = {}
        input_cols = st.columns(3)
        default_row = forecast.iloc[0].to_dict() if forecast is not None and len(forecast) > 0 else {}

        for i, feature in enumerate(feature_columns):
            col = input_cols[i % 3]
            default_value = float(default_row.get(feature, 0.0)) if feature in default_row else 0.0
            input_values[feature] = col.number_input(
                feature,
                value=default_value,
                format="%.4f"
            )

        # Better version: ask month BEFORE predict
        live_month_number = st.selectbox(
            "Month Number for Barangay Ranking",
            list(range(1, 13)),
            index=0
        )

        if st.button("Predict"):
            input_df = pd.DataFrame([input_values])

            pred = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_df)[0][1])
            else:
                prob = np.nan

            st.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
            st.info(
                f"Predicted Outbreak Probability: {prob:.4f}"
                if not pd.isna(prob) else
                "Probability not available"
            )

            st.markdown(
                """
**How to read this result**
- **0** means **Non-outbreak month**
- **1** means **Outbreak month**
- The probability is the model's estimated likelihood that the month is an outbreak month
- This is about **monthly outbreak classification**, not percentage of people or percentage of the population
"""
            )

            # Added barangay prediction inside live prediction
            st.subheader("Likely Highest-Risk Barangays")

            if barangay_risk_profile is not None and not barangay_risk_profile.empty:
                barangay_live = barangay_risk_profile.copy()

                if "overall_share" not in barangay_live.columns:
                    barangay_live["overall_share"] = 0.0
                if "recent_share" not in barangay_live.columns:
                    barangay_live["recent_share"] = 0.0

                # If seasonal_share exists in profile and month column exists, use it
                if "seasonal_share" in barangay_live.columns and "Month" in barangay_live.columns:
                    seasonal_subset = barangay_live[barangay_live["Month"] == live_month_number].copy()
                    if not seasonal_subset.empty:
                        barangay_live = seasonal_subset

                barangay_live["risk_score_raw"] = (
                    0.60 * barangay_live["recent_share"] +
                    0.40 * barangay_live["overall_share"]
                )

                total_score = barangay_live["risk_score_raw"].sum()
                if total_score > 0:
                    barangay_live["risk_score"] = barangay_live["risk_score_raw"] / total_score
                else:
                    barangay_live["risk_score"] = 0.0

                city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1 + (0 if pd.isna(prob) else prob))
                barangay_live["predicted_barangay_cases_proxy"] = barangay_live["risk_score"] * city_cases_proxy

                barangay_live_top3 = barangay_live.sort_values(
                    "predicted_barangay_cases_proxy",
                    ascending=False
                ).head(3)

                st.dataframe(barangay_live_top3, use_container_width=True)

                fig_live_barangay = px.bar(
                    barangay_live_top3,
                    x="Barangay",
                    y="predicted_barangay_cases_proxy",
                    color="Barangay",
                    title="Top 3 Likely Barangays for the Predicted Month"
                )
                st.plotly_chart(fig_live_barangay, use_container_width=True)
            else:
                st.warning("barangay_risk_profile.csv not found or empty.")

st.markdown("---")
st.caption("Baguio City Dengue Forecast Dashboard")