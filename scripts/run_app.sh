#!/bin/bash
# Disease Prediction System - MongoDB Edition
# Quick Start Script for Mac/Linux

echo ""
echo "===================================================================="
echo "   Disease Prediction System - MongoDB Authentication"
echo "===================================================================="
echo ""

# Check if virtual environment exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
else
    echo "No virtual environment found. Using system Python..."
fi

# Check if secrets.toml exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo ""
    echo "[WARNING] MongoDB configuration not found!"
    echo ""
    echo "Please run setup first:"
    echo "   python setup_mongodb.py"
    echo ""
    echo "Or create .streamlit/secrets.toml manually"
    echo ""
    exit 1
fi

# Run the application
echo ""
echo "Starting application..."
echo ""
echo "Once loaded, open your browser to: http://localhost:8501"
echo ""
echo "Demo Login:"
echo "  Username: demo_user"
echo "  Password: demo123"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""
echo "===================================================================="
echo ""

streamlit run app_mongodb.py

