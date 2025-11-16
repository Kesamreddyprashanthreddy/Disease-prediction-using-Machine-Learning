"""List all users in database"""
import sys
sys.path.insert(0, 'src')

from database import get_user_operations

# Get user operations
user_ops, db_conn = get_user_operations()

# List all users
try:
    if hasattr(user_ops, 'session'):  # SQL
        from database import User
        users = user_ops.session.query(User).all()
        print(f"Total users: {len(users)}")
        for user in users:
            print(f"  - Username: {user.username}, Email: {user.email}, Name: {user.full_name}")
    else:  # MongoDB
        users = list(user_ops.users.find({}, {'username': 1, 'email': 1, 'full_name': 1}))
        print(f"Total users: {len(users)}")
        for user in users:
            print(f"  - Username: {user['username']}, Email: {user['email']}, Name: {user.get('full_name', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")

db_conn.close()
