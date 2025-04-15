#!/usr/bin/env python
"""
YFin - Advanced Stock Tracker
Entry point script to run the application
"""

import os
import sys
import subprocess

def main():
    """Main entry point for the application"""
    # Add src directory to path
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    sys.path.insert(0, src_dir)
    
    # Run the Streamlit application
    subprocess.run(["streamlit", "run", os.path.join(src_dir, "app.py")])

if __name__ == "__main__":
    main() 