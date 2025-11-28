"""Quick script to check user in database"""
import sys
sys.path.insert(0, 'src')

from database import get_user_operations
import bcrypt

# Get user operations
user_ops, db_conn = get_user_operations()

# Check for user
username = 'prashanth'
user = user_ops.get_user_by_username(username)

if user:
    print(f"✓ User found: {username}")
    print(f"  Username: {user['username'] if isinstance(user, dict) else user.username}")
    print(f"  Email: {user['email'] if isinstance(user, dict) else user.email}")
    print(f"  Password hash: {(user['password_hash'] if isinstance(user, dict) else user.password_hash)[:50]}...")
    
    # Test password
    password = 'Reddy@123'
    stored_hash = user['password_hash'] if isinstance(user, dict) else user.password_hash
    
    print(f"\nTesting password: {password}")
    try:
        result = bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        print(f"Password verification: {'✓ MATCH' if result else '✗ NO MATCH'}")
    except Exception as e:
        print(f"Error verifying password: {e}")
else:
    print(f"✗ User not found: {username}")
    print("\nChecking for similar usernames...")
    # For MongoDB, try to list all users
    try:
        if hasattr(user_ops, 'users'):
            all_users = list(user_ops.users.find({}, {'username': 1, '_id': 0}).limit(10))
            print(f"Available users: {[u['username'] for u in all_users]}")
    except:
        pass

db_conn.close()
