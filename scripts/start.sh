#!/bin/bash
# Quick Start Script for Disease Prediction System with Authentication

echo "================================================================"
echo "🩺 Disease Prediction System - Quick Start"
echo "================================================================"
echo ""
echo "This script will help you set up the authentication system"
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "✅ .env file found"
else
    echo "⚠️  .env file not found"
    echo ""
    echo "📝 You need to create a .env file with your database URL"
    echo ""
    echo "Quick options:"
    echo "1. Run: python setup_auth.py (interactive setup)"
    echo "2. Copy .env.example to .env and edit it manually"
    echo ""
    read -p "Run interactive setup now? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        python setup_auth.py
    else
        echo "Please create .env file manually"
        exit 1
    fi
fi

echo ""
echo "📦 Checking dependencies..."

# Check if required packages are installed
python -c "import bcrypt, pymongo, dotenv, sqlalchemy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed"
else
    echo "⚠️  Some dependencies missing"
    read -p "Install dependencies now? (Y/n): " response
    if [[ ! "$response" =~ ^[Nn]$ ]]; then
        pip install -r requirements.txt
    fi
fi

echo ""
echo "================================================================"
echo "🚀 Starting Application..."
echo "================================================================"
echo ""
echo "The application will open in your browser"
echo ""
echo "📌 Quick Guide:"
echo "   1. Click 'Register' to create an account"
echo "   2. Fill in the registration form"
echo "   3. Login with your credentials"
echo "   4. Access disease prediction modules from the sidebar"
echo ""
echo "================================================================"
echo ""

# Start Streamlit
streamlit run Home.py
