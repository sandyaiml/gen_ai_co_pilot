# viz.py
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st



# Sparkline helper used by app.py KPI row
def sparkline_figure(series: pd.Series, height: int = 60) -> go.Figure:
    fig = go.Figure()
    if series is None or series.dropna().empty:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', showlegend=False))
        fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0))
        return fig
    x = list(range(len(series)))
    fig.add_trace(go.Scatter(x=x, y=series.values, mode='lines', showlegend=False))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# Re-usable SCADA trends UI (keeps your original behavior)
@st.cache_data(ttl=60)
def load_scada_timeseries(_client, scada_view: str, hours: int = 6) -> pd.DataFrame:
    ts_sql = f"""
    SELECT
      timestamp_iso,
      kiln_burning_zone_temp_C,
      coal_feed_tph,
      O2_pct,
      CO2_pct,
      kiln_speed_rpm
    FROM `{scada_view}`
    WHERE timestamp_iso >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    ORDER BY timestamp_iso ASC
    """
    df = _client.query(ts_sql).to_dataframe()
    if df.empty:
        return df
    try:
        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'])
    except Exception:
        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce')
    df = df.dropna(how='all')
    return df

def _downsample_df(df: pd.DataFrame, max_points: int = 800) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df
    tmp = df.set_index('timestamp_iso')
    total_seconds = (tmp.index.max() - tmp.index.min()).total_seconds()
    if total_seconds <= 0:
        return df.head(max_points)
    bucket_seconds = max(int(total_seconds / max_points), 1)
    rule = f"{max(1, bucket_seconds)}S"
    numeric_cols = tmp.select_dtypes(include=[np.number]).columns
    res = tmp[numeric_cols].resample(rule).mean()
    res = res.dropna(how='all').reset_index()
    if len(res) > max_points:
        idx = np.linspace(0, len(res) - 1, max_points).astype(int)
        res = res.iloc[idx].reset_index(drop=True)
    return res

def create_plotly_trend(df: pd.DataFrame, param: str, hours: int) -> go.Figure:
    if df.empty or param not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig
    plot_df = df[['timestamp_iso', param]].dropna()
    plot_df = _downsample_df(plot_df, max_points=1200)
    fig = px.line(plot_df, x='timestamp_iso', y=param, markers=False, title=f"{param} — last {hours}h")
    fig.update_layout(margin=dict(l=40, r=20, t=40, b=40), height=380)
    fig.update_xaxes(title_text="Timestamp")
    fig.update_yaxes(title_text=param, autorange=True)
    try:
        if len(plot_df) >= 5:
            plot_df['rolling'] = plot_df[param].rolling(window=max(3, int(len(plot_df)/50))).mean()
            fig.add_trace(go.Scatter(x=plot_df['timestamp_iso'], y=plot_df['rolling'],
                                     mode='lines', name='Rolling mean', line=dict(dash='dash')))
    except Exception:
        pass
    return fig

def display_scada_trends_ui(client, scada_view: str):
    st.markdown("SCADA Trends (Interactive Plot)")
    selected_hours = st.selectbox("Lookback window (hours)", options=[1,2,3,6,12,24,48], index=2, key='trend_lookback')
    with st.spinner(f"Loading SCADA timeseries (last {selected_hours} hours)..."):
        df = load_scada_timeseries(client, scada_view, hours=selected_hours)
    if df.empty:
        st.info("No SCADA timeseries rows found for the selected time window.")
        return
    params = [c for c in df.columns if c != 'timestamp_iso' and pd.api.types.is_numeric_dtype(df[c])]
    if not params:
        st.info("No numeric SCADA parameters available to plot.")
        return
    chosen_param = st.selectbox("Parameter to plot", options=params, index=0, key='trend_param')
    if st.checkbox("Show raw timeseries table (preview)", value=False, key='show_raw_table'):
        st.dataframe(df.tail(200))
    fig = create_plotly_trend(df, chosen_param, selected_hours)
    st.plotly_chart(fig, use_container_width=True)
    vals = df[chosen_param].dropna().astype(float)
    if not vals.empty:
        mean_val = vals.mean()
        std_val = vals.std()
        min_val = vals.min()
        max_val = vals.max()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_val:.3f}")
        c2.metric("Std", f"{std_val:.3f}")
        c3.metric("Min", f"{min_val:.3f}")
        c4.metric("Max", f"{max_val:.3f}")
