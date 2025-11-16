"""
🏠 AI-Powered Medical Diagnosis System - Home Page
==================================================
Multi-Disease Prediction System with Real-Time Authentication
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.auth import auth
except ImportError:
    # Fallback for different deployment environments
    from auth import auth

# Page configuration
st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🏥",
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
    
    .top-nav h1 {
        font-size: 1.3rem;
        margin: 0;
        white-space: nowrap;
    }
    
    /* Hero section - Compact */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin-bottom: 0;
    }
    
    /* Feature cards - Compact professional design */
    .feature-card {
        background: transparent;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
        height: auto;
        min-height: 160px;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
        display: inline-block;
    }
    
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .feature-description {
        color: #7f8c8d;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0.75rem;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .feature-list li {
        color: #34495e;
        font-size: 0.85rem;
        background: #f8f9fa;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .feature-list li:before {
        content: "✓";
        color: #27ae60;
        font-weight: bold;
        margin-right: 0.3rem;
    }
    
    /* Stat boxes - Compact */
    .stat-box {
        background: transparent;
        color: #2c3e50;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-3px);
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.3rem 0;
    }
    
    .stat-label {
        color: #7f8c8d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 0.9rem;
        white-space: nowrap;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Remove padding from columns on mobile */
    [data-testid="column"] {
        padding: 0 0.25rem !important;
    }
    
    /* Section headers - Compact */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin: 2rem 0 1.5rem 0;
    }
    
    /* Welcome message - Compact */
    .welcome-msg {
        background: transparent;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #ecf0f1;
    }
    
    /* Responsive Design - Mobile optimized */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem !important;
        }
        
        .top-nav {
            padding: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        .top-nav h1 {
            font-size: 1rem !important;
        }
        
        .hero-title {
            font-size: 1.4rem !important;
            line-height: 1.3 !important;
        }
        
        .hero-subtitle {
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }
        
        .hero-section {
            padding: 1.2rem 0.8rem !important;
            margin-bottom: 1.2rem !important;
        }
        
        .feature-card {
            padding: 0.9rem 1rem !important;
            margin: 0.6rem 0 !important;
            min-height: 135px !important;
        }
        
        .feature-title {
            font-size: 0.95rem !important;
            display: block !important;
            margin-left: 0 !important;
            margin-top: 0.4rem !important;
        }
        
        .feature-icon {
            font-size: 1.8rem !important;
            display: block !important;
            margin-bottom: 0.3rem !important;
        }
        
        .feature-description {
            font-size: 0.8rem !important;
            line-height: 1.4 !important;
        }
        
        .feature-list {
            gap: 0.3rem !important;
        }
        
        .feature-list li {
            font-size: 0.7rem !important;
            padding: 0.2rem 0.5rem !important;
        }
        
        .stat-box {
            padding: 0.8rem !important;
            margin-bottom: 0.6rem !important;
        }
        
        .stat-value {
            font-size: 1.3rem !important;
        }
        
        .stat-icon {
            font-size: 1.4rem !important;
        }
        
        .stat-label {
            font-size: 0.7rem !important;
        }
        
        .section-header {
            font-size: 1.2rem !important;
            margin: 1.3rem 0 0.8rem 0 !important;
        }
        
        .stButton>button {
            padding: 0.6rem 1rem !important;
            font-size: 0.85rem !important;
        }
        
        .footer {
            padding: 1.5rem 0.5rem !important;
            margin-top: 2rem !important;
        }
        
        .footer h3 {
            font-size: 1.1rem !important;
        }
        
        .footer p {
            font-size: 0.8rem !important;
            line-height: 1.4 !important;
        }
    }
    
    @media (max-width: 480px) {
        .main {
            padding: 0.3rem !important;
        }
        
        .top-nav {
            padding: 0.4rem 0.6rem !important;
        }
        
        .top-nav h1 {
            font-size: 0.9rem !important;
        }
        
        .hero-title {
            font-size: 1.2rem !important;
        }
        
        .hero-subtitle {
            font-size: 0.75rem !important;
        }
        
        .hero-section {
            padding: 1rem 0.6rem !important;
            margin-bottom: 1rem !important;
        }
        
        .feature-card {
            padding: 0.7rem 0.8rem !important;
            min-height: 125px !important;
            margin: 0.5rem 0 !important;
        }
        
        .feature-title {
            font-size: 0.9rem !important;
        }
        
        .feature-description {
            font-size: 0.75rem !important;
        }
        
        .feature-icon {
            font-size: 1.6rem !important;
        }
        
        .feature-list li {
            font-size: 0.65rem !important;
            padding: 0.15rem 0.4rem !important;
        }
        
        .section-header {
            font-size: 1.1rem !important;
            margin: 1rem 0 0.7rem 0 !important;
        }
        
        .stat-box {
            padding: 0.7rem !important;
        }
        
        .stat-value {
            font-size: 1.1rem !important;
        }
        
        .stat-icon {
            font-size: 1.2rem !important;
        }
        
        .stat-label {
            font-size: 0.65rem !important;
        }
        
        .stButton>button {
            padding: 0.5rem 0.8rem !important;
            font-size: 0.8rem !important;
        }
        
        .footer {
            padding: 1rem 0.3rem !important;
        }
        
        .footer h3 {
            font-size: 1rem !important;
        }
        
        .footer p {
            font-size: 0.7rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize authentication
auth.__init__()

# Top navigation bar - Professional Design
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1rem 2rem; 
            border-radius: 12px; 
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <span style="font-size: 2rem; margin-right: 0.5rem;">🩺</span>
        <div>
            <h1 style="margin: 0; color: white; font-size: 1.5rem; font-weight: 700;">MediAI</h1>
            <p style="margin: 0; color: rgba(255,255,255,0.9); font-size: 0.75rem; font-weight: 500;">AI-Powered Medical Diagnostics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if auth.is_authenticated():
        user_info = auth.get_current_user()
        st.markdown(f"""
        <div style="text-align: right; padding: 0.5rem 0;">
            <p style="margin: 0; color: rgba(255,255,255,0.8); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Welcome</p>
            <p style="margin: 0; color: white; font-size: 0.95rem; font-weight: 600;">{user_info['full_name']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button("Sign Up", use_container_width=True, key="nav_signup"):
            st.switch_page("pages/_Register.py")

with col3:
    if auth.is_authenticated():
        if st.button("Logout", use_container_width=True, key="nav_logout", type="secondary"):
            auth.logout_user()
            st.rerun()
    else:
        if st.button("Sign In", use_container_width=True, type="primary", key="nav_signin"):
            st.switch_page("pages/_Login.py")

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
        <span class="feature-icon">🫁</span><span class="feature-title">Lung Disease Detection</span>
        <div class="feature-description">AI-powered chest X-ray analysis for pneumonia detection</div>
        <ul class="feature-list">
            <li>Upload X-rays</li>
            <li>CNN model</li>
            <li>97% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🫁 Open Lung Disease Module", use_container_width=True, type="primary", key="lung_btn"):
        st.switch_page("pages/1_🫁_Lung_Disease.py")
    
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🫘</span><span class="feature-title">Kidney Disease Assessment</span>
        <div class="feature-description">CKD risk assessment using clinical parameters</div>
        <ul class="feature-list">
            <li>24 parameters</li>
            <li>Random Forest</li>
            <li>98% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🫘 Open Kidney Disease Module", use_container_width=True, type="primary", key="kidney_btn"):
        st.switch_page("pages/3_🫘_Kidney_Disease.py")

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🩺</span><span class="feature-title">Diabetes Prediction</span>
        <div class="feature-description">Type 2 diabetes risk using medical indicators</div>
        <ul class="feature-list">
            <li>8 parameters</li>
            <li>Random Forest</li>
            <li>95% accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🩺 Open Diabetes Module", use_container_width=True, type="primary", key="diabetes_btn"):
        st.switch_page("pages/2_🩺_Diabetes.py")
    
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎗️</span><span class="feature-title">Breast Cancer Detection</span>
        <div class="feature-description">Mammogram analysis for cancer screening</div>
        <ul class="feature-list">
            <li>Upload mammograms</li>
            <li>ResNet50</li>
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
    <p style="margin-top: 1.5rem; font-weight: 600;">🔒 Secure • 🎯 Accurate • ⚡ Fast • 🏥 Professional</p>
</div>
""", unsafe_allow_html=True)
