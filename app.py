"""
Gradio wrapper for Streamlit app - Hugging Face Spaces deployment
"""

import subprocess
import os

# Set environment for Streamlit
os.environ["STREAMLIT_SERVER_PORT"] = "7860"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

# Launch Streamlit app
if __name__ == "__main__":
    subprocess.run([
        "streamlit", "run", "Home.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.serverAddress=0.0.0.0"
    ])
