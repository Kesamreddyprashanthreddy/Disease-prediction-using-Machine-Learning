"""
📊 Prediction History Module
============================
View and analyze disease prediction history
Track patient screening records over time
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_prediction_operations
from auth import auth

# Page configuration
st.set_page_config(page_title="Prediction History", page_icon="📊", layout="wide")

# Navigation buttons
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("🏠 Home", key="home_btn", use_container_width=True):
        st.switch_page("Home.py")
with col3:
    if auth.is_authenticated():
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            auth.logout_user()
            st.switch_page("Home.py")

st.markdown("---")

# Custom CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .page-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    
    /* Record card */
    .record-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    
    .record-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .record-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .record-date {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    
    .record-result {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .result-normal {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    
    .result-disease {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .page-title {
            font-size: 1.8rem;
        }
        
        .page-subtitle {
            font-size: 0.95rem;
        }
        
        .stat-value {
            font-size: 2rem;
        }
        
        .record-title {
            font-size: 1.1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Check authentication
if not auth.check_auth():
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">📊 Prediction History</h1>
        <p class="page-subtitle">View your screening records and track health trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="auth-prompt">
        <div class="auth-prompt-icon">🔒</div>
        <h2 class="auth-prompt-title">Authentication Required</h2>
        <p class="auth-prompt-text">Please log in to view your prediction history</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔑 Go to Login", use_container_width=True, type="primary"):
            st.switch_page("pages/_Login.py")
    st.stop()

# Page header
st.markdown("""
<div class="page-header">
    <h1 class="page-title">📊 Prediction History</h1>
    <p class="page-subtitle">Track and analyze your disease screening results</p>
</div>
""", unsafe_allow_html=True)

# Fetch prediction history
try:
    pred_ops, db_conn = get_prediction_operations()
    
    # Get all predictions for the user
    all_predictions = pred_ops.get_user_predictions(
        username=st.session_state.username,
        limit=100
    )
    
    if not all_predictions:
        st.info("📭 No prediction history found. Start by analyzing your health data in any of the disease modules!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🫁 Lung Disease", use_container_width=True):
                st.switch_page("pages/1_🫁_Lung_Disease.py")
        
        with col2:
            if st.button("🩺 Diabetes", use_container_width=True):
                st.switch_page("pages/2_🩺_Diabetes.py")
        
        with col3:
            if st.button("🫘 Kidney Disease", use_container_width=True):
                st.switch_page("pages/3_🫘_Kidney_Disease.py")
        
        with col4:
            if st.button("🎗️ Breast Cancer", use_container_width=True):
                st.switch_page("pages/4_🎗️_Breast_Cancer.py")
        
        st.stop()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_predictions)
    
    # Display statistics
    st.markdown("### 📈 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(df)}</div>
            <div class="stat-label">Total Screenings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        lung_count = len(df[df['disease_type'] == 'lung'])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{lung_count}</div>
            <div class="stat-label">Lung Screenings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        diabetes_count = len(df[df['disease_type'] == 'diabetes'])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{diabetes_count}</div>
            <div class="stat-label">Diabetes Tests</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        kidney_count = len(df[df['disease_type'] == 'kidney'])
        breast_count = len(df[df['disease_type'] == 'breast_cancer'])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{kidney_count + breast_count}</div>
            <div class="stat-label">Other Screenings</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Screenings by Disease Type")
        
        disease_counts = df['disease_type'].value_counts().reset_index()
        disease_counts.columns = ['Disease Type', 'Count']
        
        # Map disease types to readable names
        disease_names = {
            'lung': '🫁 Lung Disease',
            'diabetes': '🩺 Diabetes',
            'kidney': '🫘 Kidney Disease',
            'breast_cancer': '🎗️ Breast Cancer'
        }
        disease_counts['Disease Type'] = disease_counts['Disease Type'].map(disease_names)
        
        fig_disease = px.pie(
            disease_counts,
            values='Count',
            names='Disease Type',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_disease.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_disease, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Screening Timeline")
        
        # Convert created_at to datetime if it's not already
        if 'created_at' in df.columns:
            if isinstance(df['created_at'].iloc[0], str):
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Group by date
            df['date'] = df['created_at'].dt.date
            timeline = df.groupby('date').size().reset_index(name='count')
            
            fig_timeline = px.line(
                timeline,
                x='date',
                y='count',
                markers=True,
                labels={'date': 'Date', 'count': 'Number of Screenings'}
            )
            fig_timeline.update_layout(
                xaxis_title="Date",
                yaxis_title="Screenings",
                showlegend=False
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Filters
    st.markdown("---")
    st.markdown("### 🔍 Filter Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        disease_filter = st.multiselect(
            "Disease Type",
            options=['All'] + list(df['disease_type'].unique()),
            default=['All']
        )
    
    with col2:
        result_filter = st.multiselect(
            "Result",
            options=['All'] + list(df['prediction_result'].unique()),
            default=['All']
        )
    
    with col3:
        days_filter = st.selectbox(
            "Time Period",
            options=['All Time', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days']
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'All' not in disease_filter and disease_filter:
        filtered_df = filtered_df[filtered_df['disease_type'].isin(disease_filter)]
    
    if 'All' not in result_filter and result_filter:
        filtered_df = filtered_df[filtered_df['prediction_result'].isin(result_filter)]
    
    if days_filter != 'All Time':
        days = {'Last 7 Days': 7, 'Last 30 Days': 30, 'Last 90 Days': 90}[days_filter]
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_df = filtered_df[pd.to_datetime(filtered_df['created_at']) >= cutoff_date]
    
    # Display predictions
    st.markdown("---")
    st.markdown(f"### 📋 Prediction Records ({len(filtered_df)} results)")
    
    if len(filtered_df) == 0:
        st.info("No predictions match your filter criteria.")
    else:
        # Sort by date (newest first)
        filtered_df = filtered_df.sort_values('created_at', ascending=False)
        
        # Display each record
        for idx, record in filtered_df.iterrows():
            disease_icons = {
                'lung': '🫁',
                'diabetes': '🩺',
                'kidney': '🫘',
                'breast_cancer': '🎗️'
            }
            
            disease_names = {
                'lung': 'Lung Disease',
                'diabetes': 'Diabetes',
                'kidney': 'Kidney Disease',
                'breast_cancer': 'Breast Cancer'
            }
            
            icon = disease_icons.get(record['disease_type'], '📊')
            disease_name = disease_names.get(record['disease_type'], record['disease_type'])
            
            # Determine result class
            normal_results = ['Normal', 'Low Risk', 'Benign']
            result_class = 'result-normal' if record['prediction_result'] in normal_results else 'result-disease'
            
            # Format date
            if isinstance(record['created_at'], str):
                date_obj = pd.to_datetime(record['created_at'])
            else:
                date_obj = record['created_at']
            date_str = date_obj.strftime('%B %d, %Y at %I:%M %p')
            
            with st.expander(f"{icon} {disease_name} - {record['prediction_result']} ({date_str})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Prediction Result:** {record['prediction_result']}")
                    st.markdown(f"**Confidence:** {record['confidence']:.1%}" if record['confidence'] else "**Confidence:** N/A")
                    st.markdown(f"**Disease Type:** {disease_name}")
                    st.markdown(f"**Date:** {date_str}")
                    
                    # Display test data if available
                    if record.get('test_data'):
                        test_data = record['test_data']
                        if isinstance(test_data, str):
                            try:
                                test_data = json.loads(test_data)
                            except:
                                pass
                        
                        if isinstance(test_data, dict):
                            st.markdown("**Test Parameters:**")
                            
                            # Show patient name if available
                            if 'patient_name' in test_data:
                                st.markdown(f"- Patient: {test_data['patient_name']}")
                            
                            # Show probabilities if available
                            if 'probabilities' in test_data:
                                st.markdown("- Probabilities:")
                                for key, val in test_data['probabilities'].items():
                                    st.markdown(f"  - {key}: {val:.1%}")
                
                with col2:
                    st.markdown(f"""
                    <div class="record-result {result_class}">
                        {record['prediction_result']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Export option
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📥 Export History as CSV", use_container_width=True):
            # Prepare data for export
            export_df = filtered_df.copy()
            
            # Flatten test_data column if it exists
            if 'test_data' in export_df.columns:
                export_df['test_data'] = export_df['test_data'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
                )
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    db_conn.close()

except Exception as e:
    st.error(f"Error loading prediction history: {str(e)}")
    st.info("Please make sure your database is properly configured in the .env file.")
