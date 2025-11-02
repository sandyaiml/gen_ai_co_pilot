# appv6.py
import os
import json
import io
import time
import decimal
import datetime
from typing import Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from google.cloud import bigquery
from google import genai

from vis import sparkline_figure, display_scada_trends_ui



import warnings
import sys

st.set_page_config(page_title="AI Cement Quality Co-Pilot", page_icon="jk_icon.png", layout="wide",initial_sidebar_state="expanded")
st.set_option('client.showErrorDetails', False)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================================
# ---------- CONFIG -------------
# ================================
project = "cement-opt-genai"
location = "us-central1"
GENAI_MODEL = "gemini-2.5-flash"
SCADA_VIEW = "cement-opt-genai.SCADA_SYN.view_latest_SCADA"
ROBOLAB_VIEW = "cement-opt-genai.Robolab_SYNC.view_latest_ROBOLAB"
client = bigquery.Client(project=project)


# ================================
# -------- CUSTOM CSS ------------
# ================================
st.markdown("""
    <style>
    .stApp {
        background-color: #f9fafc;
        color: #111827;
        font-family: "Inter", sans-serif;
    }
    /* Header bar */
    .header-bar {
        background-color: #1f2937;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .header-right {
        font-size: 0.95rem;
        color: #e5e7eb;
    }
    .logo {
        height: 36px;
        border-radius: 6px;
    }
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0073E6;
    }
    /* Upload card */
    .upload-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    .upload-header {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.8rem;
    }
    .ai-text-box {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 0.8rem;
        color: #111827;
        margin-top: 0.8rem;
        font-size: 0.9rem;
    }
    hr {border: none; border-top: 1px solid #e5e7eb; margin: 1.2rem 0;}
    </style>
""", unsafe_allow_html=True)

import base64

# Read logo file
with open("company_logo.png", "rb") as f:
    logo_bytes = f.read()
    logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")

# --- Center-aligned, bold title header ---
st.markdown("""
    <style>
    .header-banner {
        background-color: #2f3640;
        padding: 1.8rem 2.5rem;
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }
    .header-left {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }
    .header-center {
        text-align: center;
        font-size: 2rem;           /* slightly larger */
        font-weight: 800;          /* extra bold */
        color: white;
        letter-spacing: 0.8px;
        text-shadow: 0 0 6px rgba(255,255,255,0.25);  /* subtle glow */
    }
    .header-right {
        text-align: right;
        font-size: 1.1rem;
        color: #d1d5db;
        font-weight: 500;
    }
    .logo {
        height: 55px;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Render centered, bold header ---
st.markdown(f"""
<div class="header-banner">
    <div class="header-left">
        <img src="data:image/png;base64,{logo_b64}" class="logo">
    </div>
    <div class="header-center">
        AI Cement Quality Co-Pilot
    </div>
    <div class="header-right">
        Plant A | Kiln 1 | Live Data Feed
    </div>
</div>
""", unsafe_allow_html=True)


st.set_option('client.showErrorDetails', False)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_column_width.*")

project = "cement-opt-genai"
location = "us-central1"
GENAI_MODEL = "gemini-2.5-flash"
SCADA_VIEW = "cement-opt-genai.SCADA_SYN.view_latest_SCADA"
ROBOLAB_VIEW = "cement-opt-genai.Robolab_SYNC.view_latest_ROBOLAB"

client = bigquery.Client(project=project)

def extract_1d_profile_from_image_bytes(img_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top = int(0.08 * h); bottom = int(0.92 * h)
    left = int(0.08 * w); right = int(0.95 * w)
    crop = gray[top:bottom, left:right]
    profile = np.sum(255 - crop, axis=0).astype(float)
    profile = gaussian_filter1d(profile, sigma=3)
    return profile, crop

def detect_peaks_from_profile(profile: np.ndarray, height_factor: float = 1.5, min_distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    mean_val = float(np.mean(profile))
    height_threshold = max(mean_val * height_factor, mean_val + 1.0)
    peaks, props = find_peaks(profile, height=height_threshold, distance=min_distance)
    peak_heights = props.get("peak_heights", profile[peaks])
    return peaks, peak_heights

def estimate_phase_percentages(peaks: np.ndarray, heights: np.ndarray, top_n: int = 6) -> Dict[str, float]:
    if len(heights) == 0:
        return {}
    idx_sorted = np.argsort(heights)[::-1][:top_n]
    top_heights = heights[idx_sorted]
    areas = top_heights
    total = float(np.sum(areas))
    percentages = {}
    for i, h in enumerate(areas):
        percentages[f"peak_{i+1}"] = float(h * 100.0 / total)
    return percentages

def build_snapshot_dict(scada_agg: Dict[str, Any], robolab_latest: Dict[str, Any],
                        robolab_agg: Dict[str, Any], phases_confirmed: Dict[str, float],
                        scada_agg_str: str = None) -> Dict[str, Any]:
    snapshot = {
        "meta": {"project": project, "ts": scada_agg.get("ts_max") or time.strftime("%Y-%m-%dT%H:%M:%SZ"), "model": GENAI_MODEL},
        "scada": scada_agg,
        "scada_str": scada_agg_str,
        "robolab_latest": robolab_latest,
        "robolab_agg": robolab_agg,
        "phases": phases_confirmed
    }
    return snapshot

def call_gemini_genai_sdk(snapshot: Dict[str, Any], user_question: str = None) -> Dict[str, Any]:
    if snapshot.get("scada", {}).get("row_count", 0) < 5:
        return {"action": None, "rationale": "INSUFFICIENT DATA", "confidence": "INSUFFICIENT DATA"}
    def serializer(obj):
        if isinstance(obj, (pd.Timestamp, datetime.datetime)):
            try: return obj.isoformat()
            except: return str(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, (decimal.Decimal,)): return float(obj)
        try: return str(obj)
        except: return None
    snapshot_json = json.dumps(snapshot, ensure_ascii=False, default=serializer)

    prompt_base = f"""
You are an expert Cement Kiln Process & Quality Engineer (operator-facing).

You will receive SCADA process data, recent Robolab chemistry, and clinker phase composition as JSON.
Your job: produce a single JSON object (no extra plaintext) containing a concise, operator-actionable recommendation.

REQUIREMENTS / GUIDELINES:
- Output MUST be valid JSON only. Do NOT include any explanatory text outside the JSON.
- Provide one primary recommendation with the following fields:
  - category: one of [Burning Zone Control, Combustion & Air Control, Kiln Operation, Raw Mix Adjustment, Cooling & Clinker Quality, Data/Sensor Check]
  - action: short, measurable instruction (max 2 lines). Use % or absolute units (e.g., "increase coal feed by 4–5%", "reduce kiln speed by 0.1 rpm").
  - action_summary: 1-line plain summary suitable for dashboard display (very short).
  - rationale: 1–2 sentences linking observed data → cause → how the action addresses it (explicitly reference metric and deviation).
  - confidence: one of [HIGH, MEDIUM, LOW, INSUFFICIENT DATA].
  - confidence_justification: concise explanation why you chose that confidence (e.g., "stable 3-hour trend, low stddev", or "high variability in coal feed (σ=2.4) makes cause ambiguous", or "only 1 lab sample available").
  - expected_outcome: 1-line predicted improvement (e.g., "increase C3S by ~+8% and improve early strength").
  - severity: one of [CRITICAL, HIGH, MEDIUM, LOW] (operator priority).

- Quantitative guidance: when recommending an adjustment give a numeric target range or percent, and a short safety boundary if relevant (e.g., "increase coal feed by 4–5% (do not exceed +8%)").
- Keep every text field short and scannable. Avoid long paragraphs.

TARGET RANGES (use as reference when deciding action & rationale):
- Kiln BZ Temp: 1400–1450°C
- O₂: 2.5–3.0%
- CO₂: 22–23%
- LSF: 96–102
- SIM: 2.0–2.5
- ALM: 1.3–1.6
- C3S target: 45%
- C2S target: 25%

CONFIDENCE RULES (how to set confidence & justification):
- HIGH: consistent multi-hour SCADA trends + >=2 lab samples that support cause. Example justification: "Stable 3-hour underburning trend; LSF & chemistry stable."
- MEDIUM: partial correlation or moderate noise (e.g., some stddev in SCADA). Example: "Temp often below target but variability present (σ~50°C)."
- LOW: conflicting or high variance in SCADA variables, or multiple changing knobs (e.g., coal feed AND speed both varying). Example: "Kiln speed and coal feed fluctuate; can't isolate root cause."
- INSUFFICIENT DATA: missing or too few lab/SCADA samples (<3 rows or <2 lab samples). Example: "Only 1 Robolab sample in last hour."

LOW-DATA / FALLBACK BEHAVIOR:
- If confidence is LOW or INSUFFICIENT DATA, DO NOT output placeholders like "Raw model output returned" or "No rationale available."
- Instead supply a clear, plain-language `rationale` and a specific `confidence_justification` explaining which data is missing or noisy and what to check next (e.g., "High variation in coal feed (σ=2.4 tph); verify feeder or sample more labs.").

OUTPUT FORMAT (exact JSON example — ensure your response follows this schema):
{{
  "category": "Burning Zone Control",
  "action_summary": "Increase coal feed 4–5%",
  "action": "Increase main burner coal feed by 4–5% to raise BZ temp to ~1420–1450°C (do not exceed +8%).",
  "rationale": "BZ temp mean 1387°C (below 1400–1450°C) causing low C3S (28%); raising flame heat increases C3S formation.",
  "confidence": "HIGH",
  "confidence_justification": "SCADA shows stable under-burning across last 3 hours (low variability in coal feed and O2); Robolab samples confirm current LSF.",
  "expected_outcome": "Higher C3S and improved early strength (~+8–12% C3S expected).",
  "severity": "HIGH"
}}

ADDITIONAL RULES:
- Always reference the specific metric(s) and numeric values from the snapshot in the `rationale` (e.g., "O2 3.8% (target 2.5–3.0%)", "BZ mean 1387°C"). 
- If recommending NO ACTION, still return JSON with action set to a concise "No change" and provide a clear rationale and confidence_justification explaining why.
- If multiple plausible single-step actions exist, recommend the highest-impact action first and mention an alternate in `rationale` (still keep action as single primary step).
- Keep total output concise to enable display on an operator dashboard.

Below is the latest plant snapshot (JSON). Use it to produce the single JSON object as specified above — do NOT emit any additional text outside the JSON.
{{snapshot_json}}
"""
    prompt = prompt_base + (f"\nFollow-up question: {user_question}\n" if user_question else "")

    try:
        gen_client = genai.Client(vertexai=True, project=project, location=location)
    except Exception as e:
        return {"action": None, "rationale": f"GenAI init error: {e}", "confidence": "LOW"}

    try:
        try:
            resp = gen_client.models.generate_content(model=GENAI_MODEL, contents=prompt)
        except AttributeError:
            resp = gen_client.generate(model=GENAI_MODEL, contents=prompt, max_output_tokens=512, temperature=0.0)
    except Exception as e:
        return {"action": None, "rationale": f"GenAI request error: {e}", "confidence": "LOW"}

    text_out = None
    try:
        if hasattr(resp, "text") and resp.text:
            text_out = resp.text
        elif hasattr(resp, "outputs") and resp.outputs:
            out0 = resp.outputs[0]
            if isinstance(out0, dict):
                text_out = out0.get("content") or out0.get("text") or json.dumps(out0)
            else:
                text_out = str(out0)
        elif isinstance(resp, dict):
            if "content" in resp:
                text_out = resp["content"]
            elif "outputs" in resp and isinstance(resp["outputs"], list) and len(resp["outputs"])>0:
                o0 = resp["outputs"][0]
                text_out = o0.get("content") if isinstance(o0, dict) else str(o0)
            else:
                text_out = json.dumps(resp)
        else:
            text_out = str(resp)
    except Exception:
        text_out = str(resp)

    try:
        parsed_json = json.loads(text_out.strip())
        return parsed_json
    except Exception:
        try:
            start = text_out.find("{"); end = text_out.rfind("}")
            if start != -1 and end != -1 and end>start:
                maybe = text_out[start:end+1]
                parsed_json = json.loads(maybe)
                return parsed_json
        except Exception:
            pass
        return {"action": text_out, "rationale": "Raw model output returned", "confidence": "LOW"}

SCADA_LATEST_SQL = f"""
SELECT *
FROM `{SCADA_VIEW}`
ORDER BY timestamp_iso DESC
LIMIT 1
"""
ROBOLAB_LATEST_SQL = f"""
SELECT *
FROM `{ROBOLAB_VIEW}`
ORDER BY `To` DESC
LIMIT 1
"""
SCADA_AGG_SQL = f"""
SELECT
  MIN(timestamp_iso) AS ts_min,
  MAX(timestamp_iso) AS ts_max,
  COUNT(*) AS row_count,
  AVG(kiln_burning_zone_temp_C) AS kiln_bz_temp_mean,
  STDDEV_POP(kiln_burning_zone_temp_C) AS kiln_bz_temp_std,
  MIN(kiln_burning_zone_temp_C) AS kiln_bz_temp_min,
  MAX(kiln_burning_zone_temp_C) AS kiln_bz_temp_max,
  AVG(coal_feed_tph) AS coal_feed_mean,
  STDDEV_POP(coal_feed_tph) AS coal_feed_std,
  MIN(coal_feed_tph) AS coal_feed_min,
  MAX(coal_feed_tph) AS coal_feed_max,
  AVG(O2_pct) AS O2_mean,
  STDDEV_POP(O2_pct) AS O2_std,
  MIN(O2_pct) AS O2_min,
  MAX(O2_pct) AS O2_max,
  AVG(CO2_pct) AS CO2_mean,
  STDDEV_POP(CO2_pct) AS CO2_std,
  MIN(CO2_pct) AS CO2_min,
  MAX(CO2_pct) AS CO2_max,
  AVG(kiln_speed_rpm) AS kiln_speed_mean,
  STDDEV_POP(kiln_speed_rpm) AS kiln_speed_std,
  MIN(kiln_speed_rpm) AS kiln_speed_min,
  MAX(kiln_speed_rpm) AS kiln_speed_max
FROM `{SCADA_VIEW}`
WHERE timestamp_iso >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
"""
ROBOLAB_AGG_SQL = f"""
SELECT
  COUNT(*) as lab_count,
  AVG(SiO2_Mean) as SiO2_mean,
  STDDEV_POP(SiO2_Mean) as SiO2_std,
  MIN(SiO2_Mean) as SiO2_min,
  MAX(SiO2_Mean) as SiO2_max,
  AVG(LSF_Mean) as LSF_mean,
  STDDEV_POP(LSF_Mean) as LSF_std
FROM `{ROBOLAB_VIEW}`
WHERE `To` >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
"""
def _format_scada_agg_pretty(scada_agg: Dict[str, Any]) -> Tuple[str, str]:
    if not scada_agg:
        return "", "No SCADA aggregates available."
    one_line = ", ".join([f"{k}: {v}" for k, v in scada_agg.items()])
    groups = {
        "Timestamps": ["ts_min", "ts_max", "row_count"],
        "Kiln BZ Temp (°C)": ["kiln_bz_temp_mean", "kiln_bz_temp_std", "kiln_bz_temp_min", "kiln_bz_temp_max"],
        "Coal feed (tph)": ["coal_feed_mean", "coal_feed_std", "coal_feed_min", "coal_feed_max"],
        "Gas": ["O2_mean", "O2_std", "O2_min", "O2_max", "CO2_mean", "CO2_std", "CO2_min", "CO2_max"],
        "Kiln speed (rpm)": ["kiln_speed_mean", "kiln_speed_std", "kiln_speed_min", "kiln_speed_max"]
    }
    def fmt_val(v):
        if v is None:
            return "N/A"
        try:
            if isinstance(v, float) or (isinstance(v, (np.floating,))):
                return f"{float(v):.3f}"
            return str(v)
        except Exception:
            return str(v)
    md_lines = []
    md_lines.append("**SCADA 1-hour aggregates**")
    md_lines.append("")
    for title, keys in groups.items():
        present = [(k, scada_agg[k]) for k in keys if k in scada_agg]
        if not present:
            continue
        md_lines.append(f"**{title}**")
        for k, v in present:
            md_lines.append(f"- {k}: {fmt_val(v)}")
        md_lines.append("")
    captured = {k for keys in groups.values() for k in keys}
    extras = [(k, v) for k, v in scada_agg.items() if k not in captured]
    if extras:
        md_lines.append("**Other metrics**")
        for k, v in extras:
            md_lines.append(f"- {k}: {fmt_val(v)}")
        md_lines.append("")
    md = "\n".join(md_lines)
    return one_line, md

def load_latest_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    scada_df = client.query(SCADA_LATEST_SQL).to_dataframe()
    robolab_df = client.query(ROBOLAB_LATEST_SQL).to_dataframe()
    return scada_df, robolab_df

def load_aggregates() -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    scada_agg_df = client.query(SCADA_AGG_SQL).to_dataframe()
    scada_agg = scada_agg_df.to_dict(orient="records")[0] if not scada_agg_df.empty else {}
    if scada_agg:
        try:
            scada_agg_str = ", ".join([f"{k}: {v}" for k, v in scada_agg.items()])
        except Exception:
            scada_agg_str = json.dumps(scada_agg, default=str)
        _, scada_agg_md = _format_scada_agg_pretty(scada_agg)
    else:
        scada_agg_str = ""
        scada_agg_md = "No SCADA aggregates available."
    st.session_state['scada_agg_md'] = scada_agg_md
    robolab_latest_df = client.query(ROBOLAB_LATEST_SQL).to_dataframe()
    robolab_latest = robolab_latest_df.to_dict(orient="records")[0] if not robolab_latest_df.empty else {}
    robolab_agg_df = client.query(ROBOLAB_AGG_SQL).to_dataframe()
    robolab_agg = robolab_agg_df.to_dict(orient="records")[0] if not robolab_agg_df.empty else {}
    try:
        robolab_latest_str = ", ".join([f"{k}: {v}" for k, v in robolab_latest.items()]) if robolab_latest else ""
    except Exception:
        robolab_latest_str = json.dumps(robolab_latest, default=str) if robolab_latest else ""
    try:
        robolab_agg_str = ", ".join([f"{k}: {v}" for k, v in robolab_agg.items()]) if robolab_agg else ""
    except Exception:
        robolab_agg_str = json.dumps(robolab_agg, default=str) if robolab_agg else ""
    st.session_state['robolab_str'] = (robolab_latest_str + (" | " + robolab_agg_str if robolab_agg_str else "")).strip(" | ")
    def _dict_lines(d: Dict[str, Any]) -> List[str]:
        lines = []
        for k, v in d.items():
            try:
                if isinstance(v, float) or isinstance(v, (np.floating,)):
                    lines.append(f"- {k}: {float(v):.3f}")
                else:
                    lines.append(f"- {k}: {v}")
            except Exception:
                lines.append(f"- {k}: {v}")
        return lines
    robolab_md_lines = ["**Robolab latest snapshot**", ""]
    if robolab_latest:
        robolab_md_lines.extend(_dict_lines(robolab_latest))
    else:
        robolab_md_lines.append("No recent Robolab sample available.")
    robolab_md_lines.append("")
    robolab_md_lines.append("**Robolab 1-hour aggregates**")
    robolab_md_lines.append("")
    if robolab_agg:
        robolab_md_lines.extend(_dict_lines(robolab_agg))
    else:
        robolab_md_lines.append("No Robolab aggregates available for the last hour.")
    robolab_md = "\n".join(robolab_md_lines)
    st.session_state['robolab_md'] = robolab_md
    return scada_agg, scada_agg_str, robolab_latest, robolab_agg

TARGETS = {
    "kiln_bz_temp": (1400, 1450),
    "O2_pct": (2.5, 3.0),
    "CO2_pct": (22.0, 23.0),
    "LSF": (96, 102),
    "SIM": (2.0, 2.5),
    "ALM": (1.3, 1.6)
}

def add_phase_comment(phase: str, comment: str):
    if 'phase_comments' not in st.session_state:
        st.session_state['phase_comments'] = {}
    logs = st.session_state['phase_comments'].get(phase, [])
    logs.insert(0, {"ts": datetime.datetime.now().isoformat(), "user": "operator", "comment": comment})
    st.session_state['phase_comments'][phase] = logs

def show_kpi_row(scada_agg: Dict[str, Any], lookback_hours: int = 6):
    cols = st.columns(4)
    temp = scada_agg.get("kiln_bz_temp_mean")
    coal = scada_agg.get("coal_feed_mean")
    o2 = scada_agg.get("O2_mean")
    speed = scada_agg.get("kiln_speed_mean")
    try:
        ts_sql = f"""
        SELECT timestamp_iso, kiln_burning_zone_temp_C, coal_feed_tph, O2_pct, kiln_speed_rpm
        FROM `{SCADA_VIEW}`
        WHERE timestamp_iso >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_hours} HOUR)
        ORDER BY timestamp_iso ASC
        """
        df = client.query(ts_sql).to_dataframe()
        if not df.empty:
            df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'])
    except Exception:
        df = pd.DataFrame()
    metrics = [
        ("Kiln BZ Temp (°C)", temp, 'kiln_bz_temp', df['kiln_burning_zone_temp_C'] if not df.empty and 'kiln_burning_zone_temp_C' in df.columns else None),
        ("Coal feed (tph)", coal, None, df['coal_feed_tph'] if not df.empty and 'coal_feed_tph' in df.columns else None),
        ("O₂ (%)", o2, 'O2_pct', df['O2_pct'] if not df.empty and 'O2_pct' in df.columns else None),
        ("Kiln speed (rpm)", speed, None, df['kiln_speed_rpm'] if not df.empty and 'kiln_speed_rpm' in df.columns else None),
    ]
    for (name, val, target_key, series), col in zip(metrics, cols):
        if val is None:
            col.metric(name, "N/A")
        else:
            if target_key and target_key in TARGETS:
                lo, hi = TARGETS[target_key]
                mid = (lo + hi) / 2.0
                delta = float(val) - mid
                col.metric(name, f"{val:.2f}", delta=f"{delta:+.2f}")
            else:
                col.metric(name, f"{val:.2f}")
        try:
            fig = sparkline_figure(series)
            col.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

#st.set_page_config(page_title="AI Cement Quality Co-Pilot", layout="wide")
if "phases_confirmed" not in st.session_state:
    st.session_state["phases_confirmed"] = {}
if "detected_peaks_info" not in st.session_state:
    st.session_state["detected_peaks_info"] = {}
if "phase_comments" not in st.session_state:
    st.session_state["phase_comments"] = {}
if "last_recommendations" not in st.session_state:
    st.session_state["last_recommendations"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_scada_agg" not in st.session_state:
    st.session_state["last_scada_agg"] = {}
if "last_scada_agg_str" not in st.session_state:
    st.session_state["last_scada_agg_str"] = ""
if "last_robolab_latest" not in st.session_state:
    st.session_state["last_robolab_latest"] = {}
if "last_robolab_agg" not in st.session_state:
    st.session_state["last_robolab_agg"] = {}
if "last_snapshot" not in st.session_state:
    st.session_state["last_snapshot"] = {}
if "last_recommendation" not in st.session_state:
    st.session_state["last_recommendation"] = {}
if "last_fetch_time" not in st.session_state:
    st.session_state["last_fetch_time"] = None

st.title("AI Cement Quality Co-Pilot")

with st.sidebar:
    st.header("XRD Phase extraction")
    uploaded = st.file_uploader("Upload XRD image (png/jpg)", type=["png","jpg","jpeg"])
    HEIGHT_FACTOR = 1.6
    PEAK_MIN_DIST = 10
    height_factor = HEIGHT_FACTOR
    peak_min_dist = PEAK_MIN_DIST

    if st.session_state.get("phases_confirmed"):
        st.subheader("Saved phases")
        st.json(st.session_state["phases_confirmed"])
        for phase, logs in st.session_state.get("phase_comments", {}).items():
            st.markdown(f"**{phase} comments ({len(logs)})**")
            for l in logs[:5]:
                st.markdown(f"- {l['ts']} — {l['user']}: {l['comment']}")
        if st.button("Export phases + comments JSON"):
            payload = {"phases": st.session_state["phases_confirmed"],
                       "comments": st.session_state.get('phase_comments', {})}
            st.download_button("Download", json.dumps(payload, indent=2),
                               file_name="phases_with_comments.json",
                               mime="application/json")

    if uploaded is not None:
        img_bytes = uploaded.read()
        profile, crop_img = extract_1d_profile_from_image_bytes(img_bytes)
        peaks, heights = detect_peaks_from_profile(profile, height_factor=height_factor,
                                                   min_distance=peak_min_dist)
        percentages = estimate_phase_percentages(peaks, heights, top_n=6)
        st.session_state["detected_peaks_info"] = {
            "peaks": peaks.tolist() if hasattr(peaks, "tolist") else list(peaks),
            "heights": heights.tolist() if hasattr(heights, "tolist") else list(heights),
            "percentages": percentages
        }
        st.subheader("Profile preview")
        st.image(crop_img, use_column_width=True)

        st.markdown("### Map detected peaks to clinker phases (edit values)")
        existing = st.session_state.get("phases_confirmed", {})

        def guess(i):
            try:
                return float(list(percentages.values())[i])
            except:
                return 0.0

        c3s = st.number_input("C3S (%)", value=float(existing.get("C3S", guess(0))), step=0.1, format="%.2f", key="c3s")
        c2s = st.number_input("C2S (%)", value=float(existing.get("C2S", guess(1))), step=0.1, format="%.2f", key="c2s")
        c3a = st.number_input("C3A (%)", value=float(existing.get("C3A", guess(2))), step=0.1, format="%.2f", key="c3a")
        c4af = st.number_input("C4AF (%)", value=float(existing.get("C4AF", guess(3))), step=0.1, format="%.2f", key="c4af")

        st.markdown("Attach a comment to these phase values (optional)")
        phase_comment_text = st.text_area("Phase comment (attached to this confirm)", value="", key='phase_comment_text')

        col_norm, col_confirm = st.columns([1, 1])
        with col_norm:
            if st.button("Normalize to 100%"):
                total = (c3s + c2s + c3a + c4af) or 1.0
                st.session_state["phases_confirmed"] = {
                    "C3S": round(c3s * 100.0 / total, 2),
                    "C2S": round(c2s * 100.0 / total, 2),
                    "C3A": round(c3a * 100.0 / total, 2),
                    "C4AF": round(c4af * 100.0 / total, 2)
                }
                if phase_comment_text.strip():
                    add_phase_comment("all", phase_comment_text.strip())
                st.success("Phases normalized and saved.")
        with col_confirm:
            if st.button("Confirm phases"):
                st.session_state["phases_confirmed"] = {"C3S": float(c3s),
                                                        "C2S": float(c2s),
                                                        "C3A": float(c3a),
                                                        "C4AF": float(c4af)}
                if phase_comment_text.strip():
                    add_phase_comment("all", phase_comment_text.strip())
                st.success("Phases confirmed for use in snapshot.")

    st.markdown("---")
    st.header("Controls")
    st.write("Fetch 1-hour aggregated snapshot and recommendation.")

    if st.button("Fetch 1hr snapshot & get recommendation"):
        st.session_state["trigger_recommendation"] = True
        st.rerun()
# process trigger_recommendation (runs early so UI below reads persisted values)
if st.session_state.get("trigger_recommendation", False):
    with st.spinner("Fetching aggregates and calling Gemini..."):
        scada_agg, scada_agg_str, robolab_latest, robolab_agg = load_aggregates()
        st.session_state["last_scada_agg"] = scada_agg
        st.session_state["last_scada_agg_str"] = scada_agg_str
        st.session_state["last_robolab_latest"] = robolab_latest
        st.session_state["last_robolab_agg"] = robolab_agg

        saved_phases = st.session_state.get("phases_confirmed", {})
        st.session_state["last_fetch_time"] = datetime.datetime.now().isoformat()

        if not saved_phases:
            st.warning("Please upload and confirm XRD phases (sidebar) before generating recommendation.")
            st.session_state["trigger_recommendation"] = False
            st.rerun()
        else:
            snapshot = build_snapshot_dict(scada_agg, robolab_latest, robolab_agg, saved_phases, scada_agg_str)
            scada_rows = int(scada_agg.get("row_count", 0) or 0)
            required_fields_ok = all([
                scada_rows >= 5,
                scada_agg.get("coal_feed_mean") is not None,
                scada_agg.get("kiln_bz_temp_mean") is not None,
                saved_phases.get("C3S") is not None
            ])
            if not required_fields_ok:
                st.error("INSUFFICIENT DATA — cannot call Gemini. Ensure SCADA 1hr aggregates and confirmed phases are present.")
                st.session_state["trigger_recommendation"] = False
                st.rerun()
            else:
                rec = call_gemini_genai_sdk(snapshot)
                st.session_state["last_snapshot"] = snapshot
                st.session_state["last_recommendation"] = rec
                st.session_state["last_recommendations"].insert(0, {
                    "ts": datetime.datetime.now().isoformat(),
                    "action": rec.get("action") if isinstance(rec, dict) else rec,
                    "rationale": rec.get("rationale") if isinstance(rec, dict) else None,
                    "confidence": rec.get("confidence") if isinstance(rec, dict) else None
                })
                st.session_state["trigger_recommendation"] = False
                st.rerun()

col_main, = st.columns([1])

# 1️⃣ Live process snapshot
# --- Custom CSS for Metric Cards ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6; /* Light gray background */
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        height: 100px; /* Fixed height for uniformity */
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d; /* Muted color for label */
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #343a40; /* Dark color for value */
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate the HTML for a metric card
# NOTE: This function needs to be defined outside the st.button block.
def create_metric_card_html(label, value_key, data_dict, format_str):
    value = data_dict.get(value_key)
    # Use 'N/A' if the value is None, otherwise format the value
    display_value = f"{value:{format_str}}" if value is not None else "N/A"
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{display_value}</div>
    </div>
    """

# Helper function for timestamp cleaning (already in your original code, kept here for clarity)
def clean_ts_display(ts_str):
    if isinstance(ts_str, str):
        # Cleans string like "Timestamp('2025-11-01 17:30:00+0000', tz='UTC')"
        return ts_str.replace("Timestamp('", "").replace("', tz='UTC')", "")
    return ts_str

with col_main:
    st.header("Live Process Snapshot")
    if st.button("Fetch Latest Data"):
        scada_df, robolab_df = load_latest_data()
        if scada_df.empty or robolab_df.empty:
            st.error("No recent data available from BigQuery.")
        else:
            st.subheader("SCADA Latest")
            scada = scada_df.iloc[0].to_dict()
            timestamp_str = clean_ts_display(scada.get('timestamp_iso', 'N/A'))
            batch_id_str = scada.get('batch_id', 'N/A')
            st.markdown(f"**Batch ID:** {batch_id_str} | **Timestamp (ISO):** {timestamp_str}")
            
            # --- SCADA ROW 1: Replacing st.metric with Custom Cards ---
            cols1 = st.columns(5)
            # Col 0: BZ Temp (°C)
            cols1[0].markdown(create_metric_card_html("BZ Temp (°C)", 'kiln_burning_zone_temp_C', scada, '.2f'), unsafe_allow_html=True)
            # Col 1: Coal Feed (tph)
            cols1[1].markdown(create_metric_card_html("Coal Feed (tph)", 'coal_feed_tph', scada, '.2f'), unsafe_allow_html=True)
            # Col 2: O₂ (%)
            cols1[2].markdown(create_metric_card_html("O₂ (%)", 'O2_pct', scada, '.2f'), unsafe_allow_html=True)
            # Col 3: CO₂ (%)
            cols1[3].markdown(create_metric_card_html("CO₂ (%)", 'CO2_pct', scada, '.2f'), unsafe_allow_html=True)
            # Col 4: Kiln Speed (rpm)
            cols1[4].markdown(create_metric_card_html("Kiln Speed (rpm)", 'kiln_speed_rpm', scada, '.2f'), unsafe_allow_html=True)
            
            # --- SCADA ROW 2: Replacing st.metric with Custom Cards ---
            cols2 = st.columns(5)
            # Col 0: Raw Meal Feed (tph)
            cols2[0].markdown(create_metric_card_html("Raw Meal Feed (tph)", 'raw_meal_feed_tph', scada, '.2f'), unsafe_allow_html=True)
            # Col 1: Cooler Temp (°C)
            cols2[1].markdown(create_metric_card_html("Cooler Temp (°C)", 'cooler_temp_C', scada, '.2f'), unsafe_allow_html=True)
            # Col 2: Fan Speed (rpm)
            cols2[2].markdown(create_metric_card_html("Fan Speed (rpm)", 'fan_speed_rpm', scada, '.0f'), unsafe_allow_html=True)
            # Col 3: Kiln Load (%)
            cols2[3].markdown(create_metric_card_html("Kiln Load (%)", 'kiln_load_pct', scada, '.1f'), unsafe_allow_html=True)
            # Col 4: Material Flow (tph)
            cols2[4].markdown(create_metric_card_html("Material Flow (tph)", 'material_flow_tph', scada, '.2f'), unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("Robolab Latest")
            robo = robolab_df.iloc[0].to_dict()
            def clean_ts_display(ts_str):
                if isinstance(ts_str, str):
                    # Cleans string like "Timestamp('2025-11-01 17:30:00+0000', tz='UTC')"
                    return ts_str.replace("Timestamp('", "").replace("', tz='UTC')", "")
                return ts_str
            # --- QC INFO ROW (Timestamp and Samples) ---
            from_ts = clean_ts_display(robo.get('From', 'N/A'))
            to_ts = clean_ts_display(robo.get('To', 'N/A'))
            samples = robo.get('Samples', 'N/A')

            st.markdown(f"**Sample Period:** {from_ts} **to** {to_ts} | **Samples:** {samples}")

            # --- ROW 1: Key Quality Indices and Composition ---
            cols1 = st.columns(7) 

            # Col 0: LSF Mean
            cols1[0].markdown(create_metric_card_html("LSF Mean", 'LSF_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 1: SiO₂ Mean
            cols1[1].markdown(create_metric_card_html("SiO₂ Mean", 'SiO2_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 2: Al₂O₃ Mean
            cols1[2].markdown(create_metric_card_html("Al₂O₃ Mean", 'Al2O3_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 3: Fe₂O₃ Mean
            cols1[3].markdown(create_metric_card_html("Fe₂O₃ Mean", 'Fe2O3_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 4: CaO Mean
            cols1[4].markdown(create_metric_card_html("CaO Mean", 'CaO_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 5: MgO Mean
            cols1[5].markdown(create_metric_card_html("MgO Mean", 'MgO_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 6: LOI Mean
            cols1[6].markdown(create_metric_card_html("LOI Mean", 'LOI_Mean', robo, '.2f'), unsafe_allow_html=True)


            # --- ROW 2: Secondary Components and Statistics (7 Columns) ---
            cols2 = st.columns(7) 

            # Col 0: K₂O Mean
            cols2[0].markdown(create_metric_card_html("K₂O Mean", 'K2O_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 1: SO₃ Mean
            cols2[1].markdown(create_metric_card_html("SO₃ Mean", 'SO3_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 2: SIM Mean
            cols2[2].markdown(create_metric_card_html("SIM Mean", 'SIM_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 3: ALM Mean
            cols2[3].markdown(create_metric_card_html("ALM Mean", 'ALM_Mean', robo, '.2f'), unsafe_allow_html=True)

            # Col 4: LSF StdDev
            cols2[4].markdown(create_metric_card_html("LSF StdDev", 'LSF_StdDev', robo, '.2f'), unsafe_allow_html=True)

            # Col 5: LSF Max
            cols2[5].markdown(create_metric_card_html("LSF Max", 'LSF_Max', robo, '.2f'), unsafe_allow_html=True)

            # Col 6: LSF Min
            cols2[6].markdown(create_metric_card_html("LSF Min", 'LSF_Min', robo, '.2f'), unsafe_allow_html=True)

st.markdown("---")
# print("Files inside /app:", os.listdir("/app"))
# sys.stdout.flush()
# 2️⃣ SCADA Trend Dashboard
st.header("SCADA Trend Dashboard")
try:
    display_scada_trends_ui(client, SCADA_VIEW)
except Exception as e:
    st.error(f"SCADA trends UI error: {e}")

st.markdown("---")
st.header("Quick KPI's")
try:
    scada_agg_preview = st.session_state.get('last_scada_agg', {})
    if not scada_agg_preview:
        st.info("Press 'Fetch 1hr snapshot & get recommendation' to populate KPIs.")
    else:
        show_kpi_row(scada_agg_preview, lookback_hours=6)
except Exception:
    st.info("KPI area: no aggregates yet.")

st.markdown("---")

# 4️⃣ AI Recommendation
if st.session_state.get("last_recommendation"):
    rec = st.session_state["last_recommendation"]
    st.header("AI Recommendation (Gemini 2.5-flash)")
    st.markdown(f"**Action:** {rec.get('action','N/A')}")
    if rec.get("rationale"):
        st.markdown(f"**Rationale:** {rec.get('rationale')}")
    if rec.get("confidence"):
        st.markdown(f"**Confidence:** {rec.get('confidence')}")
    if rec.get("expected_outcome"):
        st.markdown(f"**Expected Outcome:** {rec.get('expected_outcome')}")
else:
    st.info("No recommendation yet. Use the sidebar button to generate one.")

st.markdown("---")

# 5️⃣ Operator Chat (follow-ups)
st.header("Operator Chat (Follow-Ups)")
chat_input = st.text_input("Ask follow-up question", value=st.session_state.get('chat_input', ''), key='chat_input_box')
col_send, col_clear = st.columns([1,1])
with col_send:
    if st.button("Send question to Gemini"):
        user_q = chat_input.strip()
        if not user_q:
            st.warning("Please enter a question.")
        else:
            snapshot = st.session_state.get('last_snapshot')
            if not snapshot:
                st.error("No snapshot available — request recommendation first.")
            else:
                with st.spinner("Calling Gemini for follow-up..."):
                    resp = st.session_state["last_recommendation"]
                st.session_state['chat_history'].append({'ts': datetime.datetime.now().isoformat(), 'role': 'operator', 'text': user_q})
                st.session_state['chat_history'].append({'ts': datetime.datetime.now().isoformat(), 'role': 'AI', 'text': str(resp)})
                st.session_state['chat_input'] = ''
with col_clear:
    if st.button("Clear chat history"):
        st.session_state['chat_history'] = []

if st.session_state.get('chat_history'):
    for item in st.session_state['chat_history'][-20:]:
        if item['role'] == 'operator':
            st.markdown(f"**You — {item['ts']}**: {item['text']}")
        else:
            st.markdown(f"**AI — {item['ts']}**: {item['text']}")
else:
    st.info("No chat messages yet.")

st.markdown("---")

# 6️⃣ Recent AI recommendations
st.header("Recent AI Recommendations")
if st.session_state.get("last_recommendations"):
    for rec in st.session_state["last_recommendations"][:6]:
        st.markdown(f"- **{rec['ts']}** — {rec.get('action')} — _{rec.get('confidence')}_")
else:
    st.info("No recommendations yet. Request one to populate history.")

# col_main, = st.columns([1])

# # 1️⃣ Live process snapshot
# with col_main:
#     st.header("Live process snapshot")
#     if st.button("Fetch Latest Data"):
#         scada_df, robolab_df = load_latest_data()
#         if scada_df.empty or robolab_df.empty:
#             st.error("No recent data available from BigQuery.")
#         else:
#             st.subheader("SCADA Latest")
#             scada = scada_df.iloc[0].to_dict()
#             cols = st.columns(7)
#             cols[0].metric("BZ Temp (°C)", f"{scada.get('kiln_burning_zone_temp_C', 'N/A'):.2f}" if scada.get('kiln_burning_zone_temp_C') else "N/A")
#             cols[1].metric("Coal Feed (tph)", f"{scada.get('coal_feed_tph', 'N/A'):.2f}" if scada.get('coal_feed_tph') else "N/A")
#             cols[2].metric("O₂ (%)", f"{scada.get('O2_pct', 'N/A'):.2f}" if scada.get('O2_pct') else "N/A")
#             cols[3].metric("CO₂ (%)", f"{scada.get('CO2_pct', 'N/A'):.2f}" if scada.get('CO2_pct') else "N/A")
#             cols[4].metric("Kiln Speed (rpm)", f"{scada.get('kiln_speed_rpm', 'N/A'):.2f}" if scada.get('kiln_speed_rpm') else "N/A")
#             cols[5].metric("O₂ Target Range", "2.5 – 3.0")
#             cols[6].metric("BZ Temp Target (°C)", "1400 – 1450")

#             st.subheader("Robolab Latest")
#             robo = robolab_df.iloc[0].to_dict()
#             cols = st.columns(3)
#             cols[0].metric("LSF", f"{robo.get('LSF_Mean', 'N/A'):.2f}" if robo.get('LSF_Mean') else "N/A")
#             cols[1].metric("SiO₂ (%)", f"{robo.get('SiO2_Mean', 'N/A'):.2f}" if robo.get('SiO2_Mean') else "N/A")
#             cols[2].metric("Al₂O₃ (%)", f"{robo.get('Al2O3_Mean', 'N/A'):.2f}" if robo.get('Al2O3_Mean') else "N/A")

# st.markdown("---")

# # 2️⃣ SCADA Trend Dashboard
# st.header("SCADA Trend Dashboard")
# try:
#     display_scada_trends_ui(client, SCADA_VIEW)
# except Exception as e:
#     st.error(f"SCADA trends UI error: {e}")

# st.markdown("---")

# # 3️⃣ Quick KPIs
# st.header("Quick KPIs")
# try:
#     scada_agg_preview = st.session_state.get('last_scada_agg', {})
#     if not scada_agg_preview:
#         st.info("Press 'Fetch 1hr snapshot & get recommendation' to populate KPIs.")
#     else:
#         show_kpi_row(scada_agg_preview, lookback_hours=6)
# except Exception:
#     st.info("KPI area: no aggregates yet.")

# st.markdown("---")

# # 4️⃣ AI Recommendation (Gemini 2.5-flash)
# if st.session_state.get("last_recommendation"):
#     rec = st.session_state["last_recommendation"]
#     st.subheader("AI Recommendation (Gemini 2.5-flash)")
#     action = rec.get("action") if isinstance(rec, dict) else rec
#     rationale = rec.get("rationale") if isinstance(rec, dict) else None
#     confidence = rec.get("confidence") if isinstance(rec, dict) else None
#     confidence_justification = rec.get("confidence_justification") if isinstance(rec, dict) else None
#     expected = rec.get("expected_outcome") if isinstance(rec, dict) else None
#     severity = rec.get("severity") if isinstance(rec, dict) else None

#     st.markdown(f"**Action:** {action}")
#     if rationale:
#         st.markdown(f"**Rationale:** {rationale}")
#     if confidence:
#         st.markdown(f"**Confidence:** {confidence}")
#     if confidence_justification:
#         st.markdown(f"**Confidence Justification:** {confidence_justification}")
#     if expected:
#         st.markdown(f"**Expected Outcome:** {expected}")
#     if severity:
#         st.markdown(f"**Severity:** {severity}")

#     def clean_for_json(obj):
#         if isinstance(obj, dict):
#             return {k: clean_for_json(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [clean_for_json(v) for v in obj]
#         elif isinstance(obj, (np.integer, np.floating)):
#             return obj.item()
#         elif isinstance(obj, (np.ndarray,)):
#             return obj.tolist()
#         elif isinstance(obj, (datetime.datetime, datetime.date)):
#             return obj.isoformat()
#         else:
#             return obj

#     rec_clean = clean_for_json(rec)
#     st.download_button(
#         "Download recommendation JSON",
#         json.dumps(rec_clean, indent=2),
#         file_name="recommendation.json",
#         mime="application/json"
#     )
# else:
#     st.info("No recommendation yet. Use the sidebar button to generate one.")

# st.markdown("---")

# # 5️⃣ Operator Chat (follow-ups)
# st.header("Operator Chat (follow-ups)")
# chat_input = st.text_input("Ask follow-up question", value=st.session_state.get('chat_input', ''), key='chat_input_box')
# col_send, col_clear = st.columns([1,1])
# with col_send:
#     if st.button("Send question to Gemini"):
#         user_q = chat_input.strip()
#         if not user_q:
#             st.warning("Please enter a question.")
#         else:
#             snapshot = st.session_state.get('last_snapshot')
#             if not snapshot:
#                 st.error("No snapshot available — request recommendation first.")
#             else:
#                 with st.spinner("Calling Gemini for follow-up..."):
#                     resp = call_gemini_genai_sdk(snapshot, user_question=user_q)
#                 st.session_state['chat_history'].append({'ts': datetime.datetime.now().isoformat(), 'role': 'operator', 'text': user_q})
#                 reply_text = resp.get('action') if isinstance(resp, dict) and resp.get('action') else (resp if isinstance(resp, str) else json.dumps(resp))
#                 st.session_state['chat_history'].append({'ts': datetime.datetime.now().isoformat(), 'role': 'AI', 'text': reply_text})
#                 st.session_state['chat_input'] = ''
# with col_clear:
#     if st.button("Clear chat history"):
#         st.session_state['chat_history'] = []

# if st.session_state.get('chat_history'):
#     for item in st.session_state['chat_history'][-20:]:
#         if item['role'] == 'operator':
#             st.markdown(f"**You — {item['ts']}**: {item['text']}")
#         else:
#             st.markdown(f"**AI — {item['ts']}**: {item['text']}")
# else:
#     st.info("No chat messages yet.")

# st.markdown("---")

# # 6️⃣ Recent AI recommendations
# st.subheader("Recent AI recommendations")
# if st.session_state.get("last_recommendations"):
#     for rec in st.session_state["last_recommendations"][:6]:
#         st.markdown(f"- **{rec['ts']}** — {rec.get('action')} — _{rec.get('confidence')}_")
# else:
#     st.info("No recommendations yet. Request one to populate history.")

# st.markdown("---")
# st.write("- Phase comments are stored in-session; export them using the sidebar export button.")
# st.write("- Ensure GOOGLE_APPLICATION_CREDENTIALS is set with BigQuery + Vertex/GenAI permissions.")