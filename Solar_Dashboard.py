# app.py
"""
Streamlit dashboard for AI-assisted solar MPPT project
Features:
- Real-time (simulated) monitoring: generation, consumption, SOC, export-to-grid
- Upload real data CSV (columns: timestamp, irradiance, temperature, P_pv, consumption, soc)
- Predictive maintenance page: fit degradation model, forecast, show cell life %
- Optional MQTT hook (placeholder)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import io

# Optional: MQTT (comment out if not used)
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except Exception:
    MQTT_AVAILABLE = False

st.set_page_config(
    page_title="SolarAI - Live Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Utility: Simulated data generator
# ---------------------------
@st.cache_data
def generate_initial_data(n=120, start_time=None):
    """Generate n minutes of simulated data (1-minute resolution)."""
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(minutes=n-1)
    times = [start_time + timedelta(minutes=i) for i in range(n)]
    # Simulate irradiance (W/m2) : peak midday, noise
    t = np.arange(n)
    irradiance = 600 + 300 * np.sin(np.linspace(-1.2, 1.2, n)) + np.random.normal(0, 20, n)
    irradiance = np.clip(irradiance, 0, None)
    # Temperature (C)
    temp = 25 + 3 * np.sin(np.linspace(-0.4, 0.4, n)) + np.random.normal(0, 0.5, n)
    # PV power (kW) roughly proportional to irradiance with degradation/noise
    base_pv = irradiance * 0.0012  # conversion factor to kW (tweakable)
    degradation = 1.0  # no long term decay in short sim
    pv_power = base_pv * degradation + np.random.normal(0, 0.02, n)
    pv_power = np.clip(pv_power, 0, None)
    # consumption kW
    consumption = 0.5 + 0.2 * np.sin(np.linspace(0, 3.14, n)) + np.random.normal(0, 0.05, n)
    consumption = np.clip(consumption, 0, None)
    # Battery SOC (0-100)
    soc = np.clip(50 + 10 * np.sin(np.linspace(-1, 1, n)) + np.random.normal(0, 3, n), 5, 100)
    # export to grid = max(0, pv - consumption - charge)
    export = np.clip(pv_power - consumption, 0, None)
    df = pd.DataFrame({
        "timestamp": times,
        "irradiance": irradiance,
        "temperature": temp,
        "pv_power_kW": pv_power,
        "consumption_kW": consumption,
        "soc_percent": soc,
        "export_kW": export
    })
    return df

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("SolarAI Controls")
mode = st.sidebar.radio("Mode", ["Live Monitor", "Predictive Maintenance", "Settings", "Upload Data"])
st.sidebar.markdown("---")

# MQTT placeholders
if MQTT_AVAILABLE:
    st.sidebar.success("MQTT available")
else:
    st.sidebar.info("MQTT package missing (paho-mqtt). Install if you want broker ingestion.")

# Simulation controls
with st.sidebar.expander("Simulation"):
    sim_frequency = st.selectbox("Simulation update interval (s) for demo", [1, 2, 5, 10], index=2)
    add_noise = st.checkbox("Add extra noise", value=True)

# ---------------------------
# Data source: session state
# ---------------------------
if "live_df" not in st.session_state:
    st.session_state.live_df = generate_initial_data(n=180)
    st.session_state.start_time = datetime.utcnow() - timedelta(minutes=179)

def append_simulated_point():
    """Append one simulated point (mimicking a new IoT reading)."""
    last = st.session_state.live_df.iloc[-1]
    t = last["timestamp"] + timedelta(minutes=1)
    # small random walk for variables
    irr = float(max(0, last["irradiance"] + np.random.normal(0, 10)))
    temp = float(last["temperature"] + np.random.normal(0, 0.2))
    base_pv = irr * 0.0012
    pv = float(max(0, base_pv + np.random.normal(0, 0.02)))
    cons = float(max(0, last["consumption_kW"] + np.random.normal(0, 0.03)))
    soc = float(np.clip(last["soc_percent"] + (pv - cons) * 2 + np.random.normal(0, 0.5), 0, 100))
    exp = float(max(0, pv - cons))
    new_row = {
        "timestamp": t,
        "irradiance": irr,
        "temperature": temp,
        "pv_power_kW": pv,
        "consumption_kW": cons,
        "soc_percent": soc,
        "export_kW": exp
    }
    st.session_state.live_df = pd.concat([st.session_state.live_df, pd.DataFrame([new_row])], ignore_index=True)
    # keep last 24 hours in memory (if high resolution, trim)
    st.session_state.live_df = st.session_state.live_df.iloc[-1440:].reset_index(drop=True)

# Upload data handler
uploaded_df = None
if mode == "Upload Data":
    st.header("Upload real CSV data")
    st.markdown("CSV should contain at minimum: `timestamp`, `irradiance`, `temperature`, `pv_power_kW`, `consumption_kW`, `soc_percent`")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
            st.success(f"Loaded {len(uploaded_df)} rows")
            st.dataframe(uploaded_df.head())
        except Exception as e:
            st.error("Failed to parse CSV. Ensure 'timestamp' column exists and is parseable.")
else:
    uploaded_file = None

# ---------------------------
# Live Monitor Page
# ---------------------------
if mode == "Live Monitor":
    st.title("SolarAI — Live Monitor")
    col1, col2 = st.columns((1, 1))
    df = st.session_state.live_df.copy()

    # refresh controls
    with col1:
        st.markdown("**Live update**")
        if st.button("Push new reading (simulate)"):
            append_simulated_point()
            st.rerun()
        if st.button("Append 10 points"):
            for _ in range(10):
                append_simulated_point()
            st.rerun()

    with col2:
        st.markdown("**Data source**")
        st.write("Using simulated IoT feed. Replace with real MQTT / REST when available.")
        if uploaded_file:
            st.info("Uploaded data available — switch to 'Upload Data' mode to visualize it.")

    # KPIs
    last = df.iloc[-1]
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("PV Generation (kW)", f"{last['pv_power_kW']:.3f}")
    kpi2.metric("Consumption (kW)", f"{last['consumption_kW']:.3f}")
    kpi3.metric("Battery SOC (%)", f"{last['soc_percent']:.1f}")
    kpi4.metric("Export to Grid (kW)", f"{last['export_kW']:.3f}")

    st.markdown("### Generation vs Consumption (last 3 hours)")
    # Timeseries plot
    show_df = df.copy().tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=show_df["timestamp"], y=show_df["pv_power_kW"], name="PV Generation (kW)", mode="lines"))
    fig.add_trace(go.Scatter(x=show_df["timestamp"], y=show_df["consumption_kW"], name="Consumption (kW)", mode="lines"))
    fig.update_layout(legend=dict(orientation="h"), margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SOC and Export (last 3 hours)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=show_df["timestamp"], y=show_df["soc_percent"], name="SOC %", mode="lines", yaxis="y1"))
    fig2.add_trace(go.Bar(x=show_df["timestamp"], y=show_df["export_kW"], name="Export kW", yaxis="y2", opacity=0.6))
    fig2.update_layout(yaxis=dict(title="SOC %"), yaxis2=dict(title="Export (kW)", overlaying="y", side="right"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recent readings table")
    st.dataframe(show_df.tail(20).reset_index(drop=True))

    # Download sample data
    csv = df.to_csv(index=False).encode()
    st.download_button("Download data (CSV)", csv, "solar_live_data.csv", "text/csv")

# ---------------------------
# Predictive Maintenance Page
# ---------------------------
elif mode == "Predictive Maintenance":
    st.title("Predictive Maintenance — Solar Cell Degradation")
    st.markdown("""
    This module fits a degradation model to historical *maximum daily PV power* values, then forecasts future degradation.
    - Upload historical data (timestamp, pv_power_kW).  
    - Or use the simulated dataset below (which we create by applying long-term decay to simulated daily peaks).
    """)

    # Option: upload history or use simulated history
    use_uploaded = st.checkbox("Use uploaded data (timestamp,pv_power_kW)", value=False)
    hist_df = None
    if use_uploaded and uploaded_file:
        try:
            d = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
            hist_df = d[["timestamp", "pv_power_kW"]].copy()
            st.success("Using uploaded PV history.")
        except Exception:
            st.error("Uploaded file not in expected format. Please go to Upload Data mode and try again.")
    else:
        # make simulated historical daily peaks across years
        years = st.slider("Number of years of synthetic history", 1, 10, 3)
        days = years * 365
        dates = [datetime.utcnow().date() - timedelta(days=days - i) for i in range(days)]
        # synthetic baseline peak kW and linear degradation per year
        baseline_peak = st.number_input("Baseline peak power (kW) at start", value=3.0, step=0.1)
        annual_deg_pct = st.slider("Annual degradation (%)", 0.1, 5.0, 0.7)  # typical ~0.5-1%/yr
        # build synthetic peaks
        peaks = []
        for i in range(days):
            year_frac = i / 365.0
            deg = (1 - (annual_deg_pct / 100.0) * year_frac)
            seasonal = 0.05 * np.sin(2 * np.pi * (i / 365.0))  # small yearly variation
            noise = np.random.normal(0, 0.02)
            value = max(0, baseline_peak * deg * (1 + seasonal) + noise)
            peaks.append(value)
        hist_df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "peak_power_kW": peaks
        })
        st.info("Using synthetic historical peaks for demo (replace with real history by uploading CSV).")

    st.subheader("Historical peaks (preview)")
    st.dataframe(hist_df.head())

    # Model selection
    st.subheader("Fit degradation model")
    model_type = st.selectbox("Model type", ["Linear", "Polynomial (2)", "Polynomial (3)"], index=0)
    forecast_years = st.slider("Forecast horizon (years)", 1, 10, 3)
    threshold_pct = st.slider("End-of-life threshold (% of initial)", 50, 95, 80)

    # Prepare data
    df_hist = hist_df.copy()
    # unify column names
    if "timestamp" in df_hist.columns and "pv_power_kW" in df_hist.columns:
        df_hist = df_hist.rename(columns={"timestamp": "date", "pv_power_kW": "peak_power_kW"})
    elif "date" not in df_hist.columns:
        if "timestamp" in df_hist.columns:
            df_hist["date"] = pd.to_datetime(df_hist["timestamp"])
    df_hist["date"] = pd.to_datetime(df_hist["date"]).dt.date
    # aggregate daily max (in case uploaded raw)
    df_hist = df_hist.groupby("date", as_index=False)["peak_power_kW"].max()

    # x,y
    df_hist = df_hist.sort_values("date")
    start_date = pd.to_datetime(df_hist["date"].iloc[0])
    df_hist["days_since_start"] = (pd.to_datetime(df_hist["date"]) - start_date).dt.days
    X = df_hist[["days_since_start"]].values
    y = df_hist["peak_power_kW"].values

    # choose model
    if model_type == "Linear":
        model = LinearRegression()
    elif model_type == "Polynomial (2)":
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    else:
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())

    model.fit(X, y)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.metric("Model MAE (kW)", f"{mae:.4f}")
    st.metric("Model R²", f"{r2:.4f}")

    # Forecast future days
    last_day = df_hist["days_since_start"].iloc[-1]
    future_days = int(forecast_years * 365)
    future_x = np.arange(0, last_day + future_days + 1).reshape(-1, 1)
    future_pred = model.predict(future_x)

    # compute cell life % relative to initial max
    initial_power = float(y.max())
    final_pred_days = future_pred[-1]
    life_percent = float(100.0 * future_pred[-1] / initial_power) if initial_power > 0 else 0.0

    st.markdown("### Degradation forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(df_hist["date"]), y=y, mode="markers+lines", name="Historical peaks"))
    future_dates = [start_date + timedelta(days=int(d)) for d in future_x.flatten()]
    fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines", name="Model forecast"))
    last_date = pd.to_datetime(df_hist["date"]).iloc[-1].to_pydatetime()

    # Draw the vertical line
    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color="red"
    )

    # Add a text label "Today"
    fig.add_annotation(
        x=last_date,
        y=df_hist["peak_power_kW"].max(),  # or another Y position
        text="Today",
        showarrow=False,
        yshift=10
    )

    fig.update_layout(height=450, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cell life estimates & recommendations")
    col_a, col_b = st.columns(2)
    col_a.metric("Initial peak (kW)", f"{initial_power:.3f}")
    col_b.metric(f"Predicted after {forecast_years} years (kW)", f"{future_pred[-1]:.3f}")

    st.write(f"Estimated cell life remaining at horizon: **{life_percent:.1f}%** of initial peak")
    # compute when it reaches threshold
    threshold_value = initial_power * (threshold_pct / 100.0)
    # find earliest day index where predicted <= threshold_value
    days_array = future_x.flatten()
    crossing_idx = None
    for idx, val in enumerate(future_pred):
        if val <= threshold_value:
            crossing_idx = days_array[idx]
            break
    if crossing_idx is not None:
        crossing_date = start_date + timedelta(days=int(crossing_idx))
        st.warning(f"Predicted to fall to {threshold_pct}% ({threshold_value:.3f} kW) on approx **{crossing_date.date()}**")
    else:
        st.success("Prediction does not reach the threshold within the forecast horizon.")

    # show a downloadable CSV of forecasts
    pred_df = pd.DataFrame({
        "date": future_dates,
        "predicted_peak_kW": future_pred
    })
    csv_bytes = pred_df.to_csv(index=False).encode()
    st.download_button("Download forecast CSV", csv_bytes, "degradation_forecast.csv", "text/csv")

    st.markdown("### Maintenance recommendations")
    st.write("""
    - If annual degradation exceeds ~0.8–1.0%: schedule inspection for shading, soiling, or module mismatch.  
    - If model residuals are large (MAE high) — consider re-measuring IV curves and updating model with updated datasets.  
    - If predicted EoL date is within 2–3 years, plan replacement / capacity upgrade.
    """)

# ---------------------------
# Settings Page (hooks)
# ---------------------------
elif mode == "Settings":
    st.title("Settings & IoT Hooks")
    st.markdown("Configure ingestion and model options.")

    st.subheader("MQTT Ingestion (optional)")
    st.markdown("Fill broker details to enable direct IoT ingestion. This is a placeholder — test in a secure environment.")
    broker = st.text_input("Broker host", value="", placeholder="e.g. mqtt.example.com")
    port = st.number_input("Port", value=1883)
    topic = st.text_input("Topic (subscribe)", value="solar/sensor")
    username = st.text_input("Username (optional)")
    password = st.text_input("Password (optional)", type="password")

    if st.button("Test MQTT connect"):
        if not MQTT_AVAILABLE:
            st.error("paho-mqtt not installed. Install with `pip install paho-mqtt`")
        elif not broker:
            st.error("Enter broker hostname")
        else:
            st.info("Attempting connection (demo mode). Real ingestion must be implemented in production.")
            # Minimal connection test (non-blocking, short timeout)
            try:
                client = mqtt.Client()
                if username and password:
                    client.username_pw_set(username, password)
                client.connect(broker, port, keepalive=30)
                client.disconnect()
                st.success("Connected successfully (test). Implement subscription callback to ingest messages.")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

    st.markdown("---")
    st.subheader("Model options")
    st.write("Pick default model hyperparameters (used by Predictive Maintenance).")
    st.write("Polynomial degree default: 2 (safe).")

# ---------------------------
# Fallback / Upload Data mode visual
# ---------------------------
elif mode == "Upload Data":
    st.title("Upload Data & Quick Visualize")
    st.markdown("Upload CSV with columns `timestamp, irradiance, temperature, pv_power_kW, consumption_kW, soc_percent`")
    uploaded = st.file_uploader("Upload CSV for visualization", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded, parse_dates=["timestamp"])
            df = df.sort_values("timestamp")
            st.success(f"Loaded {len(df)} rows")
            st.dataframe(df.head())
            # quick plot
            fig = px.line(df, x="timestamp", y=["pv_power_kW", "consumption_kW"], labels={"value": "kW"})
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.line(df, x="timestamp", y="soc_percent", labels={"soc_percent": "SOC %"})
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error("Failed to parse CSV. Ensure timestamp column exists and columns are named correctly.")
    else:
        st.info("No file uploaded. Use simulated live mode or upload a CSV.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Built for SolarAI research — replace simulated streams with your IoT feed or CSV uploads. Comments/questions? Add new features (MPPT setpoints, duty cycles, model serving) and I can extend this.")
