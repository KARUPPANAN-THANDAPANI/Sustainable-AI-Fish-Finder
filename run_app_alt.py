#!/usr/bin/env python3
"""
Alternative startup script for Fish Finder Application
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Fish Finder Application (Alternative Port)...")
    
    try:
        from app import app
        print("✅ Application imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Try different ports
    ports = [5000, 8000, 8080, 3000]
    
    for port in ports:
        try:
            print(f"🔄 Trying port {port}...")
            app.run(debug=False, host='127.0.0.1', port=port, threaded=False)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"❌ Port {port} busy, trying next...")
                continue
            else:
                print(f"❌ Error: {e}")
                break
    else:
        print("❌ All ports busy. Please close other applications.")

if __name__ == '__main__':
    main()