#!/usr/bin/env python3
"""
Quick script to fix indentation in the diabetes module
"""

import os

def fix_indentation(file_path, start_line, end_line):
    """Fix indentation for a specific range of lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Add 4 spaces to lines between start_line and end_line
    for i in range(start_line - 1, min(end_line, len(lines))):
        if lines[i].strip():  # Only indent non-empty lines
            lines[i] = '    ' + lines[i]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# Fix diabetes module indentation
diabetes_file = "pages/2_🩺_Diabetes.py"
if os.path.exists(diabetes_file):
    # Need to indent lines after the "if st.session_state['authentication_status']:" line
    # This should be around line 340 onwards (need to check the exact range)
    print(f"Fixing indentation for {diabetes_file}")
    fix_indentation(diabetes_file, 340, 620)  # Adjust range as needed
    print("Indentation fixed!")
else:
    print(f"File {diabetes_file} not found")
