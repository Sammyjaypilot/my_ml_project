import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import requests
import sys
import os

# Add project root to path to import ml modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page Configuration
st.set_page_config(
    page_title="Fever Severity Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .risk-medium {
        background-color: #fff4cc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffcc00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
    }
    .feature-input {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .api-status-connected {
        color: #00cc00;
        font-weight: bold;
    }
    .api-status-disconnected {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration 
API_BASE_URL = "https://fever-severity-test.onrender.com"

def check_api_health():
    """Check if the FastAPI server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    
def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None
    
def predict_with_api(patient_data):
    """Send prediction request to FastAPI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
        
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server. Please ensure your FastAPI server is running on port 8000."
    
    except requests.exceptions.Timeout:
        return None, "Request timeout. The API is taking too long to respond."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Fever Severity Assessment Tool</h1>', unsafe_allow_html=True)

    # API Status Check
    api_healthy = check_api_health()
    model_info = get_model_info() if api_healthy else None

    # Display API status
    col1, col2, col3 = st.columns(3)
    with col1:
        if api_healthy:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")

    with col2:
        if model_info:
            st.info(f"🤖 Model: {model_info.get('model_type', 'Unknown')}")

    with col3:
        if model_info:
            st.info(f"📊 Version: {model_info.get('version', 'Unknown')}")

    if not api_healthy:
        st.warning("""
        **FastAPI Server Not Detected**

        Please ensure your FastAPI server is running:
        ```bash
        uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
        ```

        The app will continue with mock predictions until the API is available.
        """)

    # Sidebar for patient information
    with st.sidebar:
        st.header("Patient Information")
        st.info("Enter the patient's symptoms and metrics to assess fever severity risk.")

        st.subheader("Vital Signs")
        temperature = st.slider("Temperature (°C)", 35.0, 42.0, 37.0, 0.1)
        heart_rate = st.slider("Heart Rate (bpm)", 50, 150, 80)
        age = st.slider("Age (years)", 0, 100, 35)

        st.subheader("Body Metrics")
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 40.0, 22.0, 0.1)
        blood_pressure = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])

        st.subheader("Environmental Factors")
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, 0.1)
        aqi = st.slider("Air Quality Index", 0, 500, 50)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        st.subheader("📊 Patient Symptoms & History")

        # Symptoms section
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            headache = st.selectbox("Headache", ["Yes", "No"])
            body_ache = st.selectbox("Body Ache", ["Yes", "No"])
            fatigue = st.selectbox("Fatigue", ["Yes", "No"])
            
        with col_s2:
            chronic_conditions = st.selectbox("Chronic Conditions", ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"])
            allergies = st.selectbox("Allergies", ["Yes", "No"])
            smoking_history = st.selectbox("Smoking History", ["Non-smoker", "Former smoker", "Current smoker"])
            alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Occasional", "Regular"])

        # Additional factors
        physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        diet_type = st.selectbox("Diet Type", ["Balanced", "Vegetarian", "Vegan", "Other"])
        previous_medication = st.selectbox("Previous Medication", ["None", "Antibiotics", "Painkillers", "Other"])
        recommended_medication = st.text_input("Recommended Medication", "Paracetamol")

        # Display current values
        current_data = {
            "Metric": ["Temperature", "Age", "Heart Rate", "BMI", "Humidity", "AQI"],
            "Value": [f"{temperature:.1f}°C", f"{age} years", f"{heart_rate} bpm", f"{bmi:.1f}",f"{humidity:.1f}%", str(aqi)]
        }

        df = pd.DataFrame(current_data)
        st.dataframe(df, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction button
        if st.button("🔍 Assess Fever Severity", type="primary", use_container_width=True):
                
            # Prepare patient data for API
            patient_data = {
                'temperature': temperature,
                'age': age,
                'bmi': bmi,
                'humidity': humidity,
                'aqi': aqi,
                'heart_rate': heart_rate,
                'gender': gender,
                'headache': headache,
                'body_ache': body_ache,
                'fatigue': fatigue,
                'chronic_conditions': chronic_conditions,
                'allergies': allergies,
                'smoking_history': smoking_history,
                'alcohol_consumption': alcohol_consumption,
                'physical_activity': physical_activity,
                'diet_type': diet_type,
                'blood_pressure': blood_pressure,
                'previous_medication': previous_medication,
                'recommended_medication': recommended_medication
            }

            with st.spinner("🔬 Analyzing health metrics with AI..."):
                if api_healthy:
                    # Use real API
                    prediction_result, error = predict_with_api(patient_data)

                    if error:
                        st.error(f"API Error: {error}")
                        st.info("Falling back to mock prediction...")
                        prediction_result = None
                else:
                    # Fallback to mock prediction
                    prediction_result = None
                    st.warning("Using mock prediction (API unavailable)")

            # Display results
            if prediction_result:
                # Real API response
                st.success("✅ AI Analysis Complete!")

                risk_level = prediction_result["risk_level"]
                probability = prediction_result["probability"]
                confidence = prediction_result.get("confidence", "High")

                # Risk visualization
                if risk_level.lower() == "high":
                    st.markdown('<div class="risk-high"><h3>🚨 High Risk Detected</h3></div>', unsafe_allow_html=True)
                    st.error("This patient shows multiple risk factors. Consider immediate medical consultation.")
                elif risk_level.lower() == "medium":
                    st.markdown('<div class="risk-medium"><h3>⚠️ Medium Risk Detected</h3></div>', unsafe_allow_html=True)
                    st.warning("Moderate risk factors present. Recommend lifestyle changes and monitoring.")
                else:
                    st.markdown('<div class="risk-low"><h3>✅ Low Risk Detected</h3></div>', unsafe_allow_html=True)
                    st.success("Patient shows minimal risk factors. Maintain current healthy lifestyle.")

                # Risk metrics from API
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Severity Level", risk_level)
                with col2:
                    st.metric("Probability", f"{probability:.1%}")
                with col3:
                    st.metric("Confidence", confidence)
                with col4:
                    st.metric("Model Version", prediction_result.get("model_version", "1.0"))

                # Show raw API response (optional, for debugging)
                with st.expander("📋 View API Response"):
                    st.json(prediction_result)
            else:
                # Mock prediction fallback for fever severity
                st.success("✅ Analysis Complete (Mock Data)")

                # Simple mock fever risk calculation
                risk_score = 0
                
                # Temperature factors
                if temperature >= 39.0: risk_score += 3
                elif temperature >= 38.0: risk_score += 2
                elif temperature >= 37.5: risk_score += 1
                
                # Age factors
                if age > 60: risk_score += 2
                elif age < 5: risk_score += 2
                elif age > 50: risk_score += 1
                
                # BMI factors
                if bmi >= 30: risk_score += 2
                elif bmi >= 25: risk_score += 1
                
                # Heart rate factors
                if heart_rate >= 100: risk_score += 2
                elif heart_rate >= 90: risk_score += 1
                
                # Symptom factors
                if headache == "Yes": risk_score += 1
                if body_ache == "Yes": risk_score += 1
                if fatigue == "Yes": risk_score += 1
                
                # Chronic conditions
                if chronic_conditions != "None": risk_score += 2
                
                # Environmental factors
                if aqi >= 150: risk_score += 1
                if humidity >= 80: risk_score += 1
                
                # Lifestyle factors
                if smoking_history == "Current smoker": risk_score += 1
                if alcohol_consumption == "Regular": risk_score += 1
                if physical_activity == "Low": risk_score += 1
                
                # Blood pressure
                if blood_pressure == "High": risk_score += 2
                elif blood_pressure == "Low": risk_score += 1

                # Determine risk level
                if risk_score >= 10:
                    risk_level = "Critical"
                    probability = 0.90
                elif risk_score >= 8:
                    risk_level = "High"
                    probability = 0.85
                elif risk_score >= 5:
                    risk_level = "Medium"
                    probability = 0.60
                else:
                    risk_level = "Low"
                    probability = 0.25

                # Display fever-specific results
                if risk_level == "Critical":
                    st.markdown('<div class="risk-high"><h3>🚨 CRITICAL FEVER RISK</h3></div>', unsafe_allow_html=True)
                    st.error("Immediate medical attention required! High fever with multiple risk factors detected.")
                elif risk_level == "High":
                    st.markdown('<div class="risk-high"><h3>🔴 High Fever Risk</h3></div>', unsafe_allow_html=True)
                    st.error("Serious fever condition. Urgent medical consultation recommended.")
                elif risk_level == "Medium":
                    st.markdown('<div class="risk-medium"><h3>🟡 Moderate Fever Risk</h3></div>', unsafe_allow_html=True)
                    st.warning("Moderate fever condition. Monitor symptoms closely and consider medical advice.")
                else:
                    st.markdown('<div class="risk-low"><h3>🟢 Low Fever Risk</h3></div>', unsafe_allow_html=True)
                    st.success("Mild fever condition. Rest and hydration recommended.")

                # Display risk metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fever Severity", risk_level)
                with col2:
                    st.metric("Risk Score", f"{risk_score}/20")
                with col3:
                    st.metric("Confidence", f"{probability:.0%}")

    with col2:
        st.subheader("💡 Connection Guide")

        if api_healthy:
            st.success("""
            **✅ API Connected Successfully**
            
            Your predictions are coming from the real ML model running on FastAPI.
            """)
        else:
            st.info("""
            **🔧 Setup Required**
            
            To use the real ML model:
            1. Start your FastAPI server
            2. Ensure it's running on port 8000
            3. Refresh this page
            
            **Current**: Using mock predictions
            """)

        st.info("""
        **📋 Fever Guidelines**
        
        **Temperature:**
        - < 37.5°C: Normal
        - 37.5-38.0°C: Mild
        - 38.1-39.0°C: Moderate
        - > 39.0°C: High
        
        **Immediate Actions:**
        - Stay hydrated
        - Monitor regularly
        - Rest adequately
        - Seek help if worsening
        """)

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Fever Severity Assessment Tool")

if __name__ == "__main__":
    main()