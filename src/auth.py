"""
Authentication module for Disease Prediction System
Handles user registration, login, password hashing, and session management
"""
import bcrypt
import streamlit as st
from datetime import datetime, timedelta
import secrets
from database import get_user_operations


class Auth:
    """Authentication handler"""
    
    def __init__(self):
        """Initialize authentication"""
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        
        # Try to restore session from persistent state
        self._restore_session()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _restore_session(self):
        """Restore session from persistent state if available"""
        # Session state persists during the Streamlit session
        # Only restore if we lost authentication but have session token
        if not st.session_state.authenticated and st.session_state.session_token:
            try:
                username = st.session_state.username
                if username:
                    # Get user from database to verify still exists
                    user_ops, db_conn = get_user_operations()
                    user = user_ops.get_user_by_username(username)
                    
                    if user:
                        if isinstance(user, dict):  # MongoDB
                            full_name = user.get('full_name', username)
                            email = user.get('email', '')
                        else:  # SQL
                            full_name = user.full_name or username
                            email = user.email
                        
                        # Restore session
                        st.session_state.authenticated = True
                        st.session_state.user_info = {
                            'username': username,
                            'full_name': full_name,
                            'email': email,
                            'login_time': datetime.now()
                        }
                    
                    db_conn.close()
            except Exception:
                pass  # Silently fail if session restoration doesn't work
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password(password: str) -> tuple[bool, str]:
        """
        Validate password strength
        Returns: (is_valid, message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        return True, "Password is strong"
    
    def register_user(self, username: str, email: str, password: str, full_name: str = None) -> tuple[bool, str]:
        """
        Register a new user
        Returns: (success, message)
        """
        try:
            # Validate inputs
            if not username or not email or not password:
                return False, "All fields are required"
            
            if len(username) < 3:
                return False, "Username must be at least 3 characters long"
            
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            is_valid, msg = self.validate_password(password)
            if not is_valid:
                return False, msg
            
            # Get database operations
            user_ops, db_conn = get_user_operations()
            
            # Check if user already exists
            if user_ops.user_exists(username=username):
                db_conn.close()
                return False, "Username already exists"
            
            if user_ops.user_exists(email=email):
                db_conn.close()
                return False, "Email already registered"
            
            # Hash password and create user
            password_hash = self.hash_password(password)
            user_ops.create_user(username, email, password_hash, full_name)
            
            db_conn.close()
            return True, "Registration successful! Please login."
        
        except Exception as e:
            return False, f"Registration error: {str(e)}"
    
    def login_user(self, username: str, password: str) -> tuple[bool, str]:
        """
        Login a user
        Returns: (success, message)
        """
        try:
            if not username or not password:
                return False, "Username and password are required"
            
            # Get database operations
            user_ops, db_conn = get_user_operations()
            
            # Get user from database
            user = user_ops.get_user_by_username(username)
            
            if not user:
                db_conn.close()
                return False, "Invalid username or password"
            
            # Get password hash based on database type
            if isinstance(user, dict):  # MongoDB
                password_hash = user['password_hash']
                full_name = user.get('full_name', username)
                email = user.get('email', '')
            else:  # SQL
                password_hash = user.password_hash
                full_name = user.full_name or username
                email = user.email
            
            # Verify password
            if not self.verify_password(password, password_hash):
                db_conn.close()
                return False, "Invalid username or password"
            
            # Update last login
            user_ops.update_last_login(username)
            
            # Generate session token
            session_token = secrets.token_urlsafe(32)
            
            # Set session state
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_info = {
                'username': username,
                'full_name': full_name,
                'email': email,
                'login_time': datetime.now()
            }
            st.session_state.session_token = session_token
            
            db_conn.close()
            return True, "Login successful!"
        
        except Exception as e:
            return False, f"Login error: {str(e)}"
    
    def logout_user(self):
        """Logout current user"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_info = None
        st.session_state.session_token = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
    def get_current_user(self) -> dict:
        """Get current user information"""
        return st.session_state.get('user_info', None)
    
    def require_auth(self):
        """Decorator-like function to require authentication"""
        if not self.is_authenticated():
            st.warning("⚠️ Please login to access this page")
            st.info("👉 Go to the Login page from the sidebar")
            st.stop()


# Create global auth instance
auth = Auth()


def show_user_info():
    """Display current user information in sidebar"""
    if auth.is_authenticated():
        user_info = auth.get_current_user()
        st.sidebar.success(f"👤 Welcome, {user_info['full_name']}!")
        st.sidebar.caption(f"Logged in as: {user_info['username']}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            auth.logout_user()
            st.rerun()
