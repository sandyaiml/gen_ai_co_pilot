import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from Recommendation_cement import generate_recommendation_with_gemini, generate_alerts_with_gemini
import base64
import plotly.express as px
import plotly.graph_objects as go

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Page Config ---
st.set_page_config(
    page_title="Cement Quality Co-Pilot",
    page_icon="jk_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Page Config ------------------
st.set_page_config(page_title="AI Cement Quality Co-Pilot", layout="wide")

# ------------------ Dummy Data Generator ------------------
@st.cache_data(ttl=30)  # refresh every 30 seconds
def generate_dummy_data(rows=24):
    np.random.seed(int(time.time()) % 10000)  
    time_index = pd.date_range(end=pd.Timestamp.now(), periods=rows, freq="H")

    data = pd.DataFrame({
        "time": time_index,
        "burning_zone_temp": np.random.randint(1250, 1500, size=rows),   # °C
        "free_lime": np.round(np.random.uniform(0.5, 3.5, size=rows), 2),  # %
        "clinker_quality": np.random.randint(0, 10, size=rows),  # 0–10 scale
        "co2_emission": np.random.randint(500, 800, size=rows)   # kg/ton
    })
    return data

df = generate_dummy_data()


def plot_line_chart(df, x_col, y_cols, title):
    fig = px.line(df, x=x_col, y=y_cols, labels={"value": "", "time": "", "variable": ""}, title=title)
    fig.update_layout(
        title=dict(x=0, font=dict(size=18, color="#333", family="Roboto")),
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center", font=dict(family="Roboto", size=11)),
        margin=dict(l=10, r=10, t=40, b=20),
        height=400,
        template="simple_white",
        font=dict(family="Roboto", size=12)
    )
    fig.update_xaxes(title_font=dict(size=14, family="Roboto"), tickfont=dict(size=10, family="Roboto"))
    fig.update_yaxes(title_font=dict(size=14, family="Roboto"), tickfont=dict(size=10, family="Roboto"))
    fig.update_traces(line=dict(width=2))
    return fig


# Create two columns: one for logo+title, one for plant info

logo_base64 = get_base64_image("company_logo.png")

# Custom CSS
st.markdown(
    f"""
    <style>
        .header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(90deg, #2c3e50, #34495e); /* dark cement tones */
            padding: 12px 25px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .logo img {{
            height: 55px;
        }}
        .title {{
            flex-grow: 1;
            text-align: center;
            font-size: 38px;
            font-weight: bold;
            color: white;
        }}
        .plant-info {{
            font-size: 18px;
            font-weight: 500;
            color: #ecf0f1;
        }}
    </style>
    <div class="header-bar">
        <div class="logo">
            <img src="data:image/png;base64,{logo_base64}">
        </div>
        <div class="title">
            AI Cement Quality Co-Pilot
        </div>
        <div class="plant-info">
            Plant A &nbsp; | &nbsp; Kiln 1 &nbsp; | &nbsp; Live Data Feed
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------ Layout ------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.subheader("Dashboard")
    latest_row = df.iloc[-1]

    # Dynamic KPI cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Burning Zone Temp", f"{latest_row['burning_zone_temp']} °C")  
    with m2:
        st.metric("Free Lime", f"{latest_row['free_lime']}%")
    with m3:
        st.metric("Clinker Quality Index", f"{latest_row['clinker_quality']}")
    with m4:
        st.metric("CO₂ Emissions", f"{latest_row['co2_emission']} kg/ton")  

 
    # Chart: Temp vs Free Lime with increased height
    st.plotly_chart(
        plot_line_chart(df, "time", ["burning_zone_temp", "free_lime"], "Burning Zone Temperature vs. Free Lime"),
        use_container_width=True
    )

# -------- Right Section (Scrollable) --------
with col2:
    with st.container(height=800): 
        st.subheader("Ask Co-Pilot")

        # -------- Case 1: Image-only Upload (auto analysis) --------
        st.markdown(
            '<div class="custom-markdown-box"><b>Case 1: Upload microscopy image for instant analysis</b></div>',
            unsafe_allow_html=True
        )
        uploaded_image_case1 = st.file_uploader(
            "Upload image for auto-analysis:",
            type=["png", "jpg", "jpeg"],
            key="image_only"
        )
        if uploaded_image_case1 is not None:
            latest_row = df.iloc[-1]
            response = generate_recommendation_with_gemini(
                temp=latest_row["burning_zone_temp"],
                free_lime=latest_row["free_lime"],
                c3s=50, c2s=25, c3a=10, c4af=15,
                microscopy_image=uploaded_image_case1,
                user_question=None
            )
            st.markdown(
                f"""<div style="padding:10px; border-radius:10px; background:#f0f0f5;
                box-shadow:0 1px 4px rgba(0,0,0,0.05); margin-top:1em;">{response}</div>""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # -------- Case 2: Image + Question (or just question) --------
        st.markdown(
            '<div class="custom-markdown-box"><b>Case 2: Ask a question (with optional microscopy image)</b></div>',
            unsafe_allow_html=True
        )
        uploaded_image_case2 = st.file_uploader(
            "Upload microscopy image (optional):",
            type=["png", "jpg", "jpeg"],
            key="image_with_question"
        )
        user_question = st.text_input("Ask about kiln performance:")

        if st.button("Get Recommendation"):
            latest_row = df.iloc[-1]
            response = generate_recommendation_with_gemini(
                temp=latest_row["burning_zone_temp"],
                free_lime=latest_row["free_lime"],
                c3s=50, c2s=25, c3a=10, c4af=15,
                microscopy_image=uploaded_image_case2,
                user_question=user_question
            )
            st.markdown(
                f"""<div style="padding:10px; border-radius:10px; background:#f0f0f5;
                box-shadow:0 1px 4px rgba(0,0,0,0.05); margin-top:1em;">{response}</div>""",
                unsafe_allow_html=True
            )

        # Clinker Quality Trend
        st.plotly_chart(
            plot_line_chart(df, "time", ["clinker_quality"], "Clinker Quality Trend (24 hrs)"),
            use_container_width=True
        )

        st.subheader("AI Insights & Alerts")
        ai_alerts = generate_alerts_with_gemini(latest_row.to_dict())

        st.markdown(
            f"""<div style="padding:10px; border-radius:8px; background:#fff3e0; border: 1px solid #ffcc80;
            box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-top:1em;">{ai_alerts}</div>""",
            unsafe_allow_html=True
        )