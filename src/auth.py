import streamlit as st
import os
import json
import hashlib
import uuid
from datetime import datetime, timedelta

# Directory for user data
USER_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'users')

def setup_auth():
    """Set up authentication in session state"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_directory():
    """Create user data directory if it doesn't exist"""
    os.makedirs(USER_DATA_DIR, exist_ok=True)

def get_user_file_path(username):
    """Get path to user file"""
    sanitized_username = username.lower().replace(' ', '_')
    return os.path.join(USER_DATA_DIR, f"{sanitized_username}.json")

def user_exists(username):
    """Check if a user exists"""
    user_file = get_user_file_path(username)
    return os.path.exists(user_file)

def save_user(username, password, email=None):
    """Save a new user"""
    create_user_directory()
    
    if user_exists(username):
        return False, "Username already exists"
    
    # Create user data
    user_data = {
        "username": username,
        "password_hash": hash_password(password),
        "user_id": str(uuid.uuid4()),
        "email": email,
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "preferences": {
            "default_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        }
    }
    
    # Save user data
    with open(get_user_file_path(username), 'w') as f:
        json.dump(user_data, f, indent=2)
    
    return True, "User created successfully"

def authenticate_user(username, password):
    """Authenticate a user"""
    if not user_exists(username):
        return False, "Username does not exist"
    
    # Load user data
    with open(get_user_file_path(username), 'r') as f:
        user_data = json.load(f)
    
    # Check password
    if user_data["password_hash"] != hash_password(password):
        return False, "Incorrect password"
    
    # Update last login time
    user_data["last_login"] = datetime.now().isoformat()
    with open(get_user_file_path(username), 'w') as f:
        json.dump(user_data, f, indent=2)
    
    # Set session state
    st.session_state['authenticated'] = True
    st.session_state['username'] = username
    st.session_state['user_id'] = user_data["user_id"]
    
    return True, "Login successful"

def logout_user():
    """Log out a user"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.session_state['user_id'] = None

def get_user_data(username):
    """Get user data"""
    if not user_exists(username):
        return None
    
    with open(get_user_file_path(username), 'r') as f:
        return json.load(f)

def save_user_preferences(username, preferences):
    """Save user preferences"""
    if not user_exists(username):
        return False
    
    user_data = get_user_data(username)
    user_data["preferences"] = preferences
    
    with open(get_user_file_path(username), 'w') as f:
        json.dump(user_data, f, indent=2)
    
    return True

def get_user_stocks(username):
    """Get stocks tracked by the user"""
    user_data = get_user_data(username)
    if not user_data:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]  # Default stocks
    
    return user_data["preferences"].get("default_stocks", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])

def set_user_stocks(username, stocks):
    """Set stocks tracked by the user"""
    if not user_exists(username):
        return False
    
    user_data = get_user_data(username)
    if "preferences" not in user_data:
        user_data["preferences"] = {}
    
    user_data["preferences"]["default_stocks"] = stocks
    
    with open(get_user_file_path(username), 'w') as f:
        json.dump(user_data, f, indent=2)
    
    return True 