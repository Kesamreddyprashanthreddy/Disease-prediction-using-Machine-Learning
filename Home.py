"""
🏠 AI-Powered Medical Diagnosis System - Home Page
==================================================
Multi-Disease Prediction System with Real-Time Authentication
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from auth import auth

# Page configuration
st.set_page_config(
    page_title="🩺 AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Top navigation bar */
    .top-nav {
        background: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Hero section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0;
    }
    
    /* Feature cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 280px;
        display: flex;
        flex-direction: column;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.8rem;
    }
    
    .feature-description {
        color: #7f8c8d;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .feature-list li {
        color: #34495e;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    
    .feature-list li:before {
        content: "✓ ";
        color: #27ae60;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Stat boxes */
    .stat-box {
        background: white;
        color: #2c3e50;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
    }
    
    .stat-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        padding: 0.75rem 2rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Section headers */
    .section-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin: 3rem 0 2rem 0;
    }
    
    /* Welcome message */
    .welcome-msg {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #ecf0f1;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem !important;
        }
        
        .hero-subtitle {
            font-size: 1rem !important;
        }
        
        .hero-section {
            padding: 2rem 1rem !important;
        }
        
        .feature-card {
            padding: 1.5rem !important;
        }
        
        .feature-title {
            font-size: 1.3rem !important;
        }
        
        .feature-description {
            font-size: 0.9rem !important;
        }
        
        .top-nav {
            padding: 0.75rem 1rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.5rem !important;
        }
        
        .hero-subtitle {
            font-size: 0.9rem !important;
        }
        
        .hero-section {
            padding: 1.5rem 0.75rem !important;
        }
        
        .feature-card {
            padding: 1rem !important;
            margin-bottom: 1rem !important;
        }
        
        .feature-title {
            font-size: 1.2rem !important;
        }
        
        .feature-description {
            font-size: 0.85rem !important;
        }
        
        .feature-icon {
            font-size: 2.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize authentication
auth.__init__()

# Top navigation bar
st.markdown('<div class="top-nav">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.markdown("# 🩺 MediAI Diagnosis")

with col2:
    if auth.is_authenticated():
        user_info = auth.get_current_user()
        st.markdown(f"**👤 {user_info['full_name']}**")
    else:
        if st.button("📝 Sign Up", use_container_width=True):
            st.switch_page("pages/_Register.py")

with col3:
    if auth.is_authenticated():
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout_user()
            st.rerun()
    else:
        if st.button("🔐 Sign In", use_container_width=True, type="primary"):
            st.switch_page("pages/_Login.py")
st.markdown('</div>', unsafe_allow_html=True)

# Hero Section
if auth.is_authenticated():
    user_info = auth.get_current_user()
    st.markdown(f"""
        <div class="hero-section">
            <div class="hero-title">Welcome back, {user_info['full_name']}! 👋</div>
            <div class="hero-subtitle">Select a disease module below to begin your AI-powered medical analysis</div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">AI-Powered Medical Diagnosis</div>
            <div class="hero-subtitle">Advanced Machine Learning for Accurate Healthcare Predictions</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<h2 class="section-header">🏥 Medical Diagnostic Modules</h2>', unsafe_allow_html=True)

# Available modules
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🫁</div>
        <div class="feature-title">Lung Disease Detection</div>
        <div class="feature-description">AI-powered chest X-ray analysis for pneumonia and tuberculosis detection</div>
        <ul class="feature-list">
            <li>Upload chest X-ray images</li>
            <li>Custom CNN model</li>
            <li>94.2% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🫁 Open Lung Disease Module", use_container_width=True, type="primary", key="lung_btn"):
        st.switch_page("pages/1_🫁_Lung_Disease.py")
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🫘</div>
        <div class="feature-title">Kidney Disease Assessment</div>
        <div class="feature-description">Chronic Kidney Disease (CKD) risk assessment using clinical parameters</div>
        <ul class="feature-list">
            <li>24 clinical parameters</li>
            <li>Random Forest model</li>
            <li>98% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🫘 Open Kidney Disease Module", use_container_width=True, type="primary", key="kidney_btn"):
        st.switch_page("pages/3_🫘_Kidney_Disease.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🩺</div>
        <div class="feature-title">Diabetes Prediction</div>
        <div class="feature-description">Type 2 diabetes risk prediction using medical indicators</div>
        <ul class="feature-list">
            <li>8 key health parameters</li>
            <li>Random Forest classifier</li>
            <li>95% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🩺 Open Diabetes Module", use_container_width=True, type="primary", key="diabetes_btn"):
        st.switch_page("pages/2_🩺_Diabetes.py")
    
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎗️</div>
        <div class="feature-title">Breast Cancer Detection</div>
        <div class="feature-description">Mammogram analysis for breast cancer screening</div>
        <ul class="feature-list">
            <li>Upload mammogram images</li>
            <li>ResNet50 transfer learning</li>
            <li>92% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎗️ Open Breast Cancer Module", use_container_width=True, type="primary", key="breast_btn"):
        st.switch_page("pages/4_🎗️_Breast_Cancer.py")


st.divider()

# Features section
st.markdown('<h2 class="section-header">✨ Platform Statistics</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-icon">🎯</div>
        <div class="stat-value">94%+</div>
        <div class="stat-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-icon">⚡</div>
        <div class="stat-value">&lt;2s</div>
        <div class="stat-label">Analysis Time</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-icon">🤖</div>
        <div class="stat-value">4</div>
        <div class="stat-label">AI Models</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-icon">📊</div>
        <div class="stat-value">PDF</div>
        <div class="stat-label">Reports</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <h3 style="color: #2c3e50; margin-bottom: 1rem;">🩺 MediAI Diagnosis System</h3>
    <p style="margin: 0.5rem 0;">Powered by Advanced Machine Learning & Deep Learning</p>
    <p style="margin: 0.5rem 0;">⚠️ For educational and research purposes only. Not a substitute for professional medical diagnosis.</p>
    <p style="margin-top: 1.5rem; font-weight: 600;">🔒 Secure • 🎯 Accurate • ⚡ Fast • 🏥 Professional</p>
</div>
""", unsafe_allow_html=True)
