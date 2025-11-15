

import streamlit as st
from PIL import Image
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib 
from fpdf import FPDF
from io import BytesIO
from streamlit_option_menu import option_menu
import plotly.express as px


st.set_page_config(page_title="AI Medical Diagnosis", page_icon="🩺", layout="wide")

MODEL_DIR = "saved_models"
LUNG_MODEL_PATH = os.path.join(MODEL_DIR, "lung_disease_model.h5")
DIABETES_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model_optimized.joblib")
DIABETES_SCALER_PATH = os.path.join(MODEL_DIR, "diabetes_scaler.joblib")
KIDNEY_MODEL_PATH = os.path.join(MODEL_DIR, "kidney_model.joblib")
KIDNEY_SCALER_PATH = os.path.join(MODEL_DIR, "kidney_scaler.joblib")
BREAST_CANCER_IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "breast_cancer_image_model.h5")


KIDNEY_FEATURE_NAMES = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

@st.cache_resource
def load_all_models():
    models = {}
    try:
        if os.path.exists(LUNG_MODEL_PATH): models['lung'] = load_model(LUNG_MODEL_PATH)
        if os.path.exists(DIABETES_MODEL_PATH): models['diabetes'] = joblib.load(DIABETES_MODEL_PATH)
        if os.path.exists(DIABETES_SCALER_PATH): models['diabetes_scaler'] = joblib.load(DIABETES_SCALER_PATH)
        if os.path.exists(KIDNEY_MODEL_PATH): models['kidney'] = joblib.load(KIDNEY_MODEL_PATH)
        if os.path.exists(KIDNEY_SCALER_PATH): models['kidney_scaler'] = joblib.load(KIDNEY_SCALER_PATH)
        if os.path.exists(BREAST_CANCER_IMAGE_MODEL_PATH): models['breast_cancer'] = load_model(BREAST_CANCER_IMAGE_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return models

models = load_all_models()


def generate_pdf_report(disease_name, prediction, confidence, image_data=None, tabular_data=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Medical Diagnosis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnosis Summary", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Analysis Type: {disease_name}", 0, 1)
    pdf.cell(0, 8, f"Prediction Result: {prediction}", 0, 1)
    if confidence is not None:
        pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", 0, 1)
    pdf.ln(5)
    
    if image_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Input Image", 0, 1)
        temp_image_path = "temp_report_image.png"
        image_data.save(temp_image_path)
        pdf.image(temp_image_path, x=pdf.get_x(), w=100)
        os.remove(temp_image_path)
        pdf.ln(5)

    if tabular_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Input Data", 0, 1)
        pdf.set_font("Arial", '', 10)
        for key, value in tabular_data.items():
            key_name = key.replace('_', ' ').title()
            pdf.cell(0, 7, f"- {key_name}: {value}", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')


def predict_lung_disease(img):
    if 'lung' not in models: st.error("Lung disease model not found."); return None, None, None
    model = models['lung']
    class_labels = ["Normal", "Pneumonia"]
    img_resized = img.convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probabilities = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100
    return predicted_class, confidence, probabilities

def predict_diabetes(input_data):
    if 'diabetes' not in models or 'diabetes_scaler' not in models: st.error("Diabetes model or scaler not found."); return None, None, None
    model = models['diabetes']
    scaler = models['diabetes_scaler']
    features = np.array([list(input_data.values())])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    predicted_class = "Diabetic" if prediction == 1 else "Not Diabetic"
    confidence = probabilities[prediction] * 100
    return predicted_class, confidence, probabilities

def predict_kidney_disease(input_data):
    if 'kidney' not in models or 'kidney_scaler' not in models: st.error("Kidney disease model or scaler not found."); return None
    model = models['kidney']
    scaler = models['kidney_scaler']
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    features_df = input_data[KIDNEY_FEATURE_NAMES]
    scaled_features = scaler.transform(features_df)
    prediction = model.predict(scaled_features)[0]
    return "Chronic Kidney Disease" if prediction == 1 else "No Chronic Kidney Disease"

def predict_breast_cancer_image(img):
    if 'breast_cancer' not in models: st.error("Breast cancer image model not found."); return None, None
    model = models['breast_cancer']
    img_resized = img.convert('RGB').resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction_prob = model.predict(img_array)[0][0]
    if prediction_prob > 0.5:
        return "Malignant", prediction_prob * 100
    else:
        return "Benign", (1 - prediction_prob) * 100

with st.sidebar:
    st.markdown("## 🩺 AI Diagnosis Suite")
    page = option_menu(
        menu_title=None,
        options=["Home", "Lung Disease", "Diabetes", "Kidney Disease", "Breast Cancer", "About"],
        icons=["house-fill", "lungs-fill", "activity", "heart-pulse-fill", "gender-female", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Overview for Clinicians & Users")
    st.markdown("""
**Lung Disease (X-Ray):**
- Deep learning (CNN) model trained to detect pneumonia and other lung conditions from chest X-rays.
- Assists radiologists by highlighting abnormal patterns, supporting faster and more accurate diagnosis.

**Diabetes (Clinical Data):**
- Machine learning model analyzes patient metrics (glucose, BMI, age, etc.) to estimate diabetes risk.
- Designed to support doctors in screening and early intervention decisions.

**Kidney Disease (Lab Results):**
- Predictive model compares patient lab values to thousands of real cases, flagging early signs of chronic kidney disease.
- Can help prioritize patients for further testing or specialist referral.

**Breast Cancer (Mammogram):**
- Advanced image model reviews mammograms for features linked to malignancy.
- Provides a second opinion for radiologists, improving confidence and workflow.

<span style='color:#e74c3c'><b>Disclaimer:</b></span> This tool is for demonstration and education only. It is not a replacement for clinical judgment or professional diagnosis.
    """, unsafe_allow_html=True)

if 'current_page' not in st.session_state or st.session_state.current_page != page:
    st.session_state.current_page = page
    keys_to_delete = [key for key in st.session_state.keys() if key.endswith('_result')]
    for key in keys_to_delete:
        del st.session_state[key]

if page == "Home":
    st.title("🩺 AI Medical Diagnosis Suite")
    st.markdown("""
Welcome to the **AI Medical Diagnosis Suite**!

This project leverages state-of-the-art machine learning models to assist in the early detection and risk assessment of four major diseases:

- **Lung Disease** (via Chest X-Ray)
- **Diabetes** (via clinical data)
- **Kidney Disease** (via lab results)
- **Breast Cancer** (via Mammogram)

Our platform enables users to upload medical images or input patient data and receive instant, AI-powered predictions and visualizations. The goal is to demonstrate how artificial intelligence can support healthcare professionals and empower patients with accessible, data-driven insights.

<span style='color:#e74c3c'><b>Note:</b></span> This tool is for educational and research purposes only. It is not intended for clinical use or as a substitute for professional medical advice.
""", unsafe_allow_html=True)

    st.markdown("""
---
### 🌍 Real-Life Applications of AI in Medical Diagnosis

Artificial Intelligence is transforming healthcare by:

- **Early Detection:** AI models can analyze X-rays and mammograms to spot early signs of lung disease and breast cancer, often before symptoms appear, enabling timely intervention.
- **Risk Assessment:** By processing patient data, AI can help identify individuals at high risk for diabetes and kidney disease, supporting preventive care and lifestyle changes.
- **Decision Support:** AI tools assist doctors by providing a second opinion, highlighting abnormal patterns, and reducing diagnostic errors.
- **Remote Screening:** In areas with limited access to specialists, AI-powered platforms enable remote screening and triage, improving healthcare accessibility.
- **Workflow Efficiency:** Automating routine image analysis and data review allows healthcare professionals to focus on complex cases and patient care.

These applications are already being piloted in hospitals, clinics, and telemedicine platforms worldwide, helping to improve outcomes and optimize healthcare resources.
""")

elif page == "Lung Disease":
    st.header("Lung Disease Prediction (X-Ray)")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"], key="lung_uploader")
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-Ray", use_container_width=True)
            if st.button("Analyze X-Ray", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    pred, conf, probs = predict_lung_disease(img)
                    if pred: st.session_state['lung_result'] = {'pred': pred, 'conf': conf, 'probs': probs, 'img': img}
    with col2:
        if 'lung_result' in st.session_state:
            with st.container(border=True):
                res = st.session_state['lung_result']
                st.metric(label="Prediction Result", value=res['pred'], delta=f"{res['conf']:.2f}% Confidence")
                prob_df = pd.DataFrame({'Condition': ["Normal", "Pneumonia"], 'Probability': res['probs']})
                fig = px.bar(prob_df, x='Condition', y='Probability', title='Prediction Probability Distribution', text=[f'{p:.1%}' for p in res['probs']], color='Condition', color_discrete_map={'Normal':'#2ecc71', 'Pneumonia':'#f39c12'})
                fig.update_layout(yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)
                pdf_bytes = generate_pdf_report("Lung Disease", res['pred'], res['conf'], image_data=res['img'])
                st.download_button("📄 Download PDF Report", pdf_bytes, "lung_report.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Upload an image and click 'Analyze' to see the results here.")

elif page == "Diabetes":
    st.header("Diabetes Prediction")
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.form("diabetes_form"):
            st.write("Enter the patient's information below.")
            pregnancies = st.number_input("Pregnancies", 0, 20, 1); glucose = st.number_input("Glucose", 0, 200, 120);
            blood_pressure = st.number_input("Blood Pressure", 0, 130, 70); skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80); bmi = st.number_input("BMI", 0.0, 70.0, 32.0, format="%.1f");
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, format="%.3f"); age = st.number_input("Age", 0, 120, 30)
            if st.form_submit_button("Predict Diabetes", use_container_width=True, type="primary"):
                st.session_state.current_page = page
                input_data = {'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,'DiabetesPedigreeFunction': dpf, 'Age': age}
                with st.spinner("Analyzing..."):
                    pred, conf, probs = predict_diabetes(input_data)
                    if pred: st.session_state['diabetes_result'] = {'pred': pred, 'conf': conf, 'probs': probs, 'data': input_data}
    with col2:
        if 'diabetes_result' in st.session_state:
            with st.container(border=True):
                res = st.session_state['diabetes_result']
                st.metric(label="Prediction Result", value=res['pred'], delta=f"{res['conf']:.2f}% Confidence")
                prob_df = pd.DataFrame({'Condition': ["Not Diabetic", "Diabetic"], 'Probability': res['probs']})
                fig = px.bar(prob_df, x='Condition', y='Probability', title='Prediction Probability Distribution', text=[f'{p:.1%}' for p in res['probs']], color='Condition', color_discrete_map={'Not Diabetic':'#2ecc71', 'Diabetic':'#e74c3c'})
                fig.update_layout(yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)
                pdf_bytes = generate_pdf_report("Diabetes", res['pred'], res['conf'], tabular_data=res['data'])
                st.download_button("📄 Download PDF Report", pdf_bytes, "diabetes_report.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Enter patient data and click 'Predict' to see the results here.")

elif page == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])
    with tab1:
        with st.form("kidney_form"):
            st.write("Enter the patient's lab results below.")
            st.subheader("Patient Vitals"); c1,c2,c3,c4=st.columns(4); age=c1.number_input("Age (age)",1,100,50); bp=c2.number_input("Blood Pressure (bp)",40,180,80); sg=c3.number_input("Specific Gravity (sg)",1.000,1.030,1.015,format="%.3f"); al=c4.number_input("Albumin (al)",0,5,1)
            st.subheader("Urine Analysis"); c1,c2,c3,c4,c5=st.columns(5); su=c1.number_input("Sugar (su)",0,5,0); rbc=c2.number_input("Red Blood Cells (rbc)",0,1,1); pc=c3.number_input("Pus Cell (pc)",0,1,1); pcc=c4.number_input("Pus Cell Clumps (pcc)",0,1,1); ba=c5.number_input("Bacteria (ba)",0,1,1)
            st.subheader("Blood Analysis"); c1,c2,c3=st.columns(3); bgr=c1.number_input("Blood Glucose Random (bgr)",20,500,120); bu=c1.number_input("Blood Urea (bu)",1,400,40); sc=c1.number_input("Serum Creatinine (sc)",0.1,80.0,1.2,format="%.1f"); sod=c2.number_input("Sodium (sod)",4,190,138); pot=c2.number_input("Potassium (pot)",2.0,50.0,4.5,format="%.1f"); hemo=c2.number_input("Haemoglobin (hemo)",3.0,18.0,15.0,format="%.1f"); pcv=c3.number_input("Packed Cell Volume (pcv)",9,55,45); wc=c3.number_input("White Blood Cell Count (wc)",2000,27000,7500); rc=c3.number_input("Red Blood Cell Count (rc)",2.0,8.0,5.2,format="%.1f")
            st.subheader("Medical History"); c1,c2,c3,c4,c5,c6=st.columns(6); htn=c1.number_input("Hypertension (htn)",0,1,0); dm=c2.number_input("Diabetes Mellitus (dm)",0,1,0); cad=c3.number_input("Coronary Artery Disease (cad)",0,1,0); appet=c4.number_input("Appetite (appet)",0,1,1); pe=c5.number_input("Pedal Edema (pe)",0,1,0); ane=c6.number_input("Anemia (ane)",0,1,0)
            if st.form_submit_button("Predict Kidney Disease", use_container_width=True, type="primary"):
                st.session_state.current_page = page
                input_data_dict = {'age':age,'bp':bp,'sg':sg,'al':al,'su':su,'rbc':rbc,'pc':pc,'pcc':pcc,'ba':ba,'bgr':bgr,'bu':bu,'sc':sc,'sod':sod,'pot':pot,'hemo':hemo,'pcv':pcv,'wc':wc,'rc':rc,'htn':htn,'dm':dm,'cad':cad,'appet':appet,'pe':pe,'ane':ane}
                with st.spinner("Analyzing..."):
                    pred = predict_kidney_disease(input_data_dict)
                    if pred: st.session_state['kidney_result'] = {'pred': pred, 'data': input_data_dict}
        if 'kidney_result' in st.session_state:
            with st.container(border=True):
                res = st.session_state['kidney_result']
                st.metric(label="Prediction Result", value=res['pred'])
                pdf_bytes = generate_pdf_report("Kidney Disease", res['pred'], None, tabular_data=res['data'])
                st.download_button("📄 Download PDF Report", pdf_bytes, "kidney_report.pdf", "application/pdf", use_container_width=True)
    with tab2:
        st.write("Upload a CSV file with patient data. Columns must match the template.")
        sample_df = pd.DataFrame([[0.0] * len(KIDNEY_FEATURE_NAMES)], columns=KIDNEY_FEATURE_NAMES)
        st.download_button("Download CSV Template", sample_df.to_csv(index=False).encode('utf-8'), "kidney_template.csv", "text/csv", key="kidney_template")
        uploaded_csv = st.file_uploader("Upload patient CSV", type=["csv"], key="kidney_uploader")
        if uploaded_csv:
            patient_df = pd.read_csv(uploaded_csv)
            st.write("Uploaded Data Preview:", patient_df.head())
            if st.button("Predict from CSV", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    pred = predict_kidney_disease(patient_df.head(1))
                    if pred: st.metric(label="Prediction Result", value=pred)

elif page == "Breast Cancer":
    st.header("Breast Cancer Prediction (Mammogram)")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload a Mammogram Image", type=["jpg", "png", "jpeg", "pgm"], key="bc_uploader")
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Mammogram", use_container_width=True)
            if st.button("Analyze Mammogram", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    pred, conf = predict_breast_cancer_image(img)
                    if pred: st.session_state['breast_cancer_result'] = {'pred': pred, 'conf': conf, 'img': img}
    with col2:
        if 'breast_cancer_result' in st.session_state:
            with st.container(border=True):
                res = st.session_state['breast_cancer_result']
                st.metric(label="Prediction Result", value=res['pred'], delta=f"{res['conf']:.2f}% Confidence")
                prob_malignant = res['conf'] / 100 if res['pred'] == 'Malignant' else (100 - res['conf']) / 100
                prob_benign = 1 - prob_malignant
                prob_df = pd.DataFrame({'Condition': ["Benign", "Malignant"], 'Probability': [prob_benign, prob_malignant]})
                fig = px.bar(prob_df, x='Condition', y='Probability', title='Prediction Probability Distribution', text=[f'{p:.1%}' for p in [prob_benign, prob_malignant]], color='Condition', color_discrete_map={'Benign':'#2ecc71', 'Malignant':'#e74c3c'})
                fig.update_layout(yaxis_range=[0,1])
                st.plotly_chart(fig, use_container_width=True)
                pdf_bytes = generate_pdf_report("Breast Cancer", res['pred'], res['conf'], image_data=res['img'])
                st.download_button("📄 Download PDF Report", pdf_bytes, "breast_cancer_report.pdf", "application/pdf", use_container_width=True)
        else:
            st.info("Upload an image and click 'Analyze' to see the results here.")

elif page == "About":
    st.header("About 'Transforming Disease Analysis with ML'")
    st.markdown("""
**This project was developed by: Nithin, Prashanth, Mahesh, and Jaswanth.**
It was created in partial fulfillment for the award of the Degree of Bachelor of Technology in Computer Science & Engineering.
    """)
    st.subheader("Models Used (2025 Update)")
    st.markdown("""
**Lung Disease:** Convolutional Neural Network (CNN) trained on chest X-ray images (`lung_disease_model.h5`)

**Diabetes:** Optimized machine learning model (see code for details, currently using `diabetes_model_optimized.joblib`)

**Kidney Disease:** Machine learning model (see code for details, currently using `kidney_model.joblib`)

**Breast Cancer:** Deep learning model for mammogram images (`breast_cancer_image_model.h5`)

*Note: Model architectures and training details are available in the project code and documentation. The models may be updated as new data or techniques become available.*
    """)