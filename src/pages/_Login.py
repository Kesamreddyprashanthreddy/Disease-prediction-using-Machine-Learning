"""
Login Page for Disease Prediction System
"""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth import auth, show_user_info

st.set_page_config(
    page_title="Login - Disease Prediction",
    page_icon="🔐",
    layout="centered"
)

# Custom CSS for centered modern design
st.markdown("""
    <style>
    .main {
        max-width: 500px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
    }
    .login-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .logo-text {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #667eea;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Show user info in sidebar if logged in
show_user_info()

# Main content
st.markdown('<div class="logo-text">🔐 Welcome Back</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sign in to access your account</div>', unsafe_allow_html=True)

# If already logged in, show success message
if auth.is_authenticated():
    st.success("✅ You are already logged in!")
    user_info = auth.get_current_user()
    
    st.info(f"""
    **User Information:**
    - **Username:** {user_info['username']}
    - **Full Name:** {user_info['full_name']}
    - **Email:** {user_info['email']}
    """)
    
    if st.button("🚪 Logout", type="primary", use_container_width=True):
        auth.logout_user()
        st.success("Logged out successfully!")
        st.rerun()
    
    st.stop()

# Login form
with st.form("login_form"):
    username = st.text_input(
        "Username",
        placeholder="Enter your username",
        help="Your registered username"
    )
    
    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        help="Your account password"
    )
    
    st.write("")
    submit_button = st.form_submit_button("🔓 Sign In", use_container_width=True, type="primary")
    
    if submit_button:
        if username and password:
            with st.spinner("Authenticating..."):
                success, message = auth.login_user(username, password)
                
                if success:
                    st.success(message)
                    
                    # Check if there's a redirect page stored
                    if 'redirect_after_login' in st.session_state:
                        redirect_page = st.session_state['redirect_after_login']
                        del st.session_state['redirect_after_login']
                        st.switch_page(redirect_page)
                    else:
                        st.switch_page("Home.py")
                else:
                    st.error(message)
        else:
            st.warning("⚠️ Please enter both username and password")

st.write("")
st.markdown("---")

# Link to registration
st.markdown("<p style='text-align: center;'>Don't have an account?</p>", unsafe_allow_html=True)
if st.button("📝 Create New Account", use_container_width=True):
    st.switch_page("pages/_Register.py")
