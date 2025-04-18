import streamlit as st
import os
import json
import hashlib
import uuid
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import multiprocessing.shared_memory as shm

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

def logout_user(rerun=False):
    """Log out a user"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.session_state['user_id'] = None
    
    if rerun:
        st.rerun()

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

def get_stock_data_02_via_ipc(shared_mem_seg_name, save_to_csv=True):
    """
    Fetch stock data from a shared memory segment created by a C++ process
    
    Args:
        shared_mem_seg_name (str): Name of the shared memory segment
        
    Returns:
        pandas.DataFrame: Stock data with the same format as get_stock_data()
    """
    try:
        # Access the existing shared memory block by name
        existing_shm = shm.SharedMemory(name=shared_mem_seg_name)
        
        # First 4 bytes (int32) contain the number of rows
        num_rows = np.ndarray(shape=(1,), dtype=np.int32, buffer=existing_shm.buf[0:4])[0]
        
        # Next 4 bytes contain the number of columns 
        num_cols = np.ndarray(shape=(1,), dtype=np.int32, buffer=existing_shm.buf[4:8])[0]
        
        # Calculate the offset for the actual data
        data_offset = 8  # 2 integers (4 bytes each)
        
        # Create a NumPy array that references the shared memory
        # Assuming data is stored as float64 (8 bytes per value)
        data_size = num_rows * num_cols
        data_array = np.ndarray(
            shape=(num_rows, num_cols),
            dtype=np.float64,
            buffer=existing_shm.buf[data_offset:]
        )
        
        # Create a DataFrame with the appropriate columns
        column_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # If the number of columns doesn't match our expected format, adjust
        if num_cols != len(column_names):
            column_names = [f'Col_{i}' for i in range(num_cols)]
        
        # Create DataFrame from the NumPy array
        df = pd.DataFrame(data_array, columns=column_names)
        
        # Generate dates for the index (assuming daily data, newest first)
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(num_rows)]
        dates.reverse()  # Oldest first
        
        # Add Date column
        df['Date'] = dates
        
        # Clean up the shared memory reference (doesn't delete the shared memory)
        existing_shm.close()
        
        # After creating the DataFrame, save to CSV
        if save_to_csv and not df.empty:
            csv_path = os.path.join(USER_DATA_DIR, f"{shared_mem_seg_name}_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Data saved to {csv_path}")
        
        return df
        
    except Exception as e:
        print(f"Error accessing shared memory: {e}")
        return pd.DataFrame()

# For parallel processing
def process_multiple_tickers(ticker_list):
    import concurrent.futures
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(ticker_list))) as executor:
        future_to_ticker = {
            executor.submit(get_stock_data_02_via_ipc, f"shm_{ticker}"): ticker 
            for ticker in ticker_list
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    return results 