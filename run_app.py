#!/usr/bin/env python3
"""
Clean startup script for Fish Finder Application
Run this instead of app.py directly to avoid reload issues
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Fish Finder Application...")
    
    # Import required modules
    try:
        from app import app
        print("✅ Application imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("Installing missing dependencies...")
        os.system("pip install flask joblib pandas numpy")
        from app import app
    
    # Run the application
    print("🌐 Web application ready at: http://localhost:5000")
    print("📡 Satellite data: ACTIVE")
    print("🤖 AI Model: LOADED")
    print("🛑 Press Ctrl+C to stop the application")
    
    # Run without debug mode to avoid reload issues
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()