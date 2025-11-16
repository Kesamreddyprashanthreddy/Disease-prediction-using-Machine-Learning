"""
Entry point for Render deployment
"""

import subprocess
import os

# Get port from environment (Render provides this)
port = os.environ.get("PORT", "10000")

# Ensure we're in the correct directory
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# Launch Streamlit app
if __name__ == "__main__":
    subprocess.run([
        "streamlit", "run", "Home.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.serverAddress=0.0.0.0",
        "--browser.gatherUsageStats=false"
    ])
