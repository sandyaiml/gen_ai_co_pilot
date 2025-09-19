import os
import mimetypes
from dotenv import load_dotenv
import google.generativeai as genai

# load_dotenv("API_KEY.env")

# api_key = os.getenv("GOOGLE_API_KEY")   
api_key = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=api_key)

def generate_recommendation_with_gemini(temp, free_lime, c3s, c2s, c3a, c4af, microscopy_image=None, user_question=None):
    model = genai.GenerativeModel("gemini-1.5-flash")

    clinker_details = f"""
    🔹 Burning Zone Temperature: {temp} °C
    🔹 Free Lime: {free_lime} %
    🔹 C3S: {c3s} %
    🔹 C2S: {c2s} %
    🔹 C3A: {c3a} %
    🔹 C4AF: {c4af} %
    """

    # Case 1: Parameters + User Question (no image)
    if microscopy_image is None and user_question:
        prompt = f"""
        You are an expert Cement Kiln Quality Engineer Assistant. 

        Analyze the clinker parameters and answer the user's question.

        {clinker_details}

        User Question: {user_question}

        Please provide a short professional recommendation (2–3 sentences).
        """
        response = model.generate_content(prompt)
        return response.text

    # Case 2: Parameters + Image + User Question
    elif microscopy_image is not None and user_question:
        prompt = f"""
        You are an expert Cement Kiln Quality Engineer Assistant. 

        Analyze both the clinker parameters and microscopy image to answer the user's question.

        {clinker_details}

        User Question: {user_question}

        Please provide a short professional recommendation (2–3 sentences).
        """

        # Handle microscopy image
        if isinstance(microscopy_image, str):
            with open(microscopy_image, "rb") as f:
                image_bytes = f.read()
            mime_type, _ = mimetypes.guess_type(microscopy_image)
        else:
            image_bytes = microscopy_image.read()
            mime_type = "image/png"

        gemini_image = {"mime_type": mime_type or "image/png", "data": image_bytes}
        response = model.generate_content([prompt, gemini_image])
        return response.text

    # Case 3: Parameters + Image (no user question)
    elif microscopy_image is not None and not user_question:
        prompt =f"""
        You are an expert Cement Kiln Quality Engineer Assistant. Answer the question asked by the user based on
        Analyze the clinker parameters and microscopy image to provide a professional recommendation.

        {clinker_details}

        Please:
        1. Identify if the clinker is underburnt, overburnt, or normal.
        2. Suggest corrective kiln actions.
        3. Keep the answer short (2–3 sentences).
        """

        if isinstance(microscopy_image, str):
            with open(microscopy_image, "rb") as f:
                image_bytes = f.read()
            mime_type, _ = mimetypes.guess_type(microscopy_image)
        else:
            image_bytes = microscopy_image.read()
            mime_type = "image/png"

        gemini_image = {"mime_type": mime_type or "image/png", "data": image_bytes}
        response = model.generate_content([prompt, gemini_image])
        return response.text

    else:
        return "No valid input (provide at least a user question or an image)."
    
def generate_alerts_with_gemini(latest_row: dict):
    model = genai.GenerativeModel("gemini-1.5-flash")

    alerts_prompt = f"""
    You are an expert AI cement process assistant.
    Analyze the latest kiln parameters and generate 2-3 short alerts with actionable solutions.

    Parameters:
    - Burning Zone Temperature: {latest_row['burning_zone_temp']} °C
    - Free Lime: {latest_row['free_lime']} %
    - Clinker Quality Index: {latest_row['clinker_quality']}
    - CO₂ Emission: {latest_row['co2_emission']} kg/ton clinker

    Guidelines for generating alerts and solutions:
    - Always include both the problem AND a short corrective action.
    - Use <div style="color:red;"> for critical issues.
    - Use <div style="color:orange;"> for warnings or suboptimal conditions.
    - Use <div style="color:green;"> for stable conditions.
    - Keep messages concise and action-oriented (max 1 line each).

    Alerts and Solutions:
    - If Burning Zone Temperature is significantly high (e.g., > 1480°C), suggest reducing fuel rate or kiln feed.
    - If Burning Zone Temperature is significantly low (e.g., < 1400°C), suggest increasing fuel rate or kiln feed.
    - If Free Lime is high (> 1.5%), suggest adjusting the raw mix to **reduce lime saturation factor (LSF)** or **increase burning zone temperature**.
    - If Free Lime is low (< 0.5%), suggest adjusting the raw mix to **increase lime saturation factor (LSF)**.
    - If the Clinker Quality Index is suboptimal (< 7), suggest reviewing **raw mix homogeneity** and **optimizing kiln atmosphere** (e.g., adjusting draft or oxygen levels) or adjusting **raw meal fineness**.
    - If CO₂ emissions are high (> 800 kg/ton), suggest **optimizing the raw mix** to reduce the use of high-calcium materials or **improving thermal efficiency** to reduce fuel consumption.
    - If all parameters are within optimal ranges, suggest maintaining current settings.

    Example outputs:
    <div style="color:red;"> High burning zone temp – Reduce fuel rate immediately.</div>
    <div style="color:orange;"> High Free Lime (2.81%) detected – Adjust raw mix to reduce LSF or increase burning zone temperature.</div>
    <div style="color:orange;"> Clinker Quality Index (6) is suboptimal – Review raw material homogeneity and optimize kiln atmosphere.</div>
    <div style="color:green;"> Burning Zone Temperature (1451°C) stable – Maintain current settings.</div>
    """

    response = model.generate_content(alerts_prompt)
    return response.text if response else "<div style='color:gray;'>No alerts generated.</div>"


