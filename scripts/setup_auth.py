"""
Quick Setup Script for Authentication System
Helps configure the database and generate required keys
"""

import secrets
import os
from pathlib import Path

def generate_secret_key():
    """Generate a secure secret key."""
    return secrets.token_urlsafe(32)

def setup_env_file():
    """Create or update .env file with necessary configuration."""
    env_path = Path('.env')
    
    print("🔧 Setting up authentication system...")
    print()
    
    # Check if .env already exists
    if env_path.exists():
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Setup cancelled.")
            return
    
    print("\n📊 Choose your database:")
    print("1. MongoDB (Recommended for cloud/free hosting)")
    print("2. PostgreSQL (Recommended for production)")
    print("3. MySQL")
    
    db_choice = input("\nEnter choice (1-3): ").strip()
    
    database_url = ""
    
    if db_choice == "1":
        print("\n📝 MongoDB Setup")
        # print("Get your connection string from: https://www.mongodb.com/cloud/atlas")
        # print("Format: mongodb+srv://username:password@cluster.mongodb.net/database_name")
        database_url = input("\nEnter MongoDB connection string: ").strip()
    
    elif db_choice == "2":
        print("\n📝 PostgreSQL Setup")
        host = input("Host (default: localhost): ").strip() or "localhost"
        port = input("Port (default: 5432): ").strip() or "5432"
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        database = input("Database name: ").strip()
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    
    elif db_choice == "3":
        print("\n📝 MySQL Setup")
        host = input("Host (default: localhost): ").strip() or "localhost"
        port = input("Port (default: 3306): ").strip() or "3306"
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        database = input("Database name: ").strip()
        database_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    
    else:
        print("❌ Invalid choice!")
        return
    
    # Generate secret key
    secret_key = generate_secret_key()
    
    # Create .env content
    env_content = f"""# Database Configuration
DATABASE_URL={database_url}

# Secret Key for Session Management
SECRET_KEY={secret_key}

# Application Settings
APP_NAME=Disease Prediction System
DEBUG=False
"""
    
    # Write to file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("\n✅ .env file created successfully!")
    print("\n🔑 Generated Secret Key (saved in .env):")
    print(f"   {secret_key}")
    print("\n📝 Database URL configured:")
    print(f"   {database_url[:30]}..." if len(database_url) > 30 else f"   {database_url}")
    print("\n⚠️  IMPORTANT: Never commit the .env file to version control!")
    
def test_database_connection():
    """Test database connection."""
    print("\n🔍 Testing database connection...")
    
    try:
        from database import DatabaseConnection
        
        db_conn = DatabaseConnection()
        connection = db_conn.connect()
        print("✅ Database connection successful!")
        db_conn.close()
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check if database server is running")
        print("   2. Verify database URL in .env file")
        print("   3. Check firewall settings")
        print("   4. Verify database credentials")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing required packages...")
    
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install some packages")
        print("💡 Try manually: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("🔐 Disease Prediction System - Authentication Setup")
    print("=" * 60)
    print()
    
    # Check if running in correct directory
    if not Path('requirements.txt').exists():
        print("❌ Error: requirements.txt not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Step 1: Setup .env file
    setup_env_file()
    
    # Step 2: Ask about installing dependencies
    print("\n" + "=" * 60)
    install = input("\n📦 Install/update dependencies now? (Y/n): ").strip().lower()
    if install != 'n':
        if install_dependencies():
            # Step 3: Test database connection
            print("\n" + "=" * 60)
            test = input("\n🔍 Test database connection? (Y/n): ").strip().lower()
            if test != 'n':
                if test_database_connection():
                    print("\n" + "=" * 60)
                    print("✅ Setup completed successfully!")
                    print("\n🚀 Next steps:")
                    print("   1. Run: streamlit run Home.py")
                    print("   2. Register a new account")
                    print("   3. Start using the disease prediction system")
                    print("=" * 60)
                else:
                    print("\n⚠️  Setup completed but database connection failed")
                    print("Please check your database configuration and try again")
    else:
        print("\n⚠️  Remember to install dependencies:")
        print("   pip install -r requirements.txt")
    
    print("\n📖 For detailed instructions, see: AUTHENTICATION_SETUP.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        print("Please check AUTHENTICATION_SETUP.md for manual setup instructions")
