import streamlit as st
import pandas as pd
import plotly.express as px
import random
from datetime import datetime, timedelta

# ---------- Fake live data generator (replace with real IoT / DB) ----------
def get_live_data(n=60):
    """Simulates live household solar data for the past n minutes"""
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i) for i in range(n)][::-1]
    pv_power = [max(0, 500 + random.uniform(-100, 100)) for _ in range(n)]  # ~500 W panel
    consumption = [max(100, 400 + random.uniform(-150, 150)) for _ in range(n)]  # 250–550 W
    battery_soc = random.randint(20, 100)
    grid_flow = [con - pv for pv, con in zip(pv_power, consumption)]  # +ve = import, -ve = export

    df = pd.DataFrame({
        "timestamp": timestamps,
        "pv_power_W": pv_power,
        "consumption_W": consumption,
        "grid_flow_W": grid_flow
    })
    return df, battery_soc

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Home Solar Monitor", layout="wide")

st.title("🏡 Household Solar Monitor")

# Get live data
df, battery_soc = get_live_data()
last = df.iloc[-1]

# ---------- Top KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("☀️ Solar Power", f"{last['pv_power_W']:.0f} W")
col2.metric("🏠 Home Usage", f"{last['consumption_W']:.0f} W")
col3.metric("🔋 Battery", f"{battery_soc} %")

if last['grid_flow_W'] > 0:
    col4.metric("⚡ From Grid", f"{last['grid_flow_W']:.0f} W")
else:
    col4.metric("⚡ To Grid", f"{abs(last['grid_flow_W']):.0f} W")

st.markdown("---")

# ---------- Graph: Solar vs Consumption ----------
st.subheader("📊 Solar vs Home Usage (Watts)")
fig = px.line(df, x="timestamp",
              y=["pv_power_W", "consumption_W"],
              labels={"value": "Watts", "variable": "Power"},
              color_discrete_map={
                  "pv_power_W": "gold",
                  "consumption_W": "steelblue"
              })
fig.update_layout(legend_title="", yaxis_title="Watts", xaxis_title="Time")
st.plotly_chart(fig, use_container_width=True)

# ---------- Battery Status ----------
st.subheader("🔋 Battery Status")
st.progress(int(battery_soc))

if battery_soc > 70:
    st.success("Battery is well charged ✅")
elif battery_soc > 30:
    st.warning("Battery at medium level ⚠️")
else:
    st.error("Battery low! Consider reducing usage ❗")

# ---------- Simple Message ----------
if last['pv_power_W'] > last['consumption_W']:
    st.info("☀️ Your solar is covering your needs and exporting to the grid.")
else:
    st.warning("⚡ Using extra power from the grid.")
