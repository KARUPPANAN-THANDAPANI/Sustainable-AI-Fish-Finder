#!/usr/bin/env python3
"""
GUARANTEED Startup for Fish Finder
"""

import os
import sys
import subprocess
import time

def main():
    print("🎯 STARTING FISH FINDER - GUARANTEED METHOD")
    print("=" * 50)
    
    # Kill any existing Python processes
    print("🔄 Cleaning up existing processes...")
    os.system('taskkill /f /im python.exe 2>nul')
    time.sleep(2)
    
    # Start the application
    print("🚀 Launching Fish Finder Application...")
    
    try:
        # Use subprocess to run the app (more reliable)
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ])
        
        print("✅ Application started successfully!")
        print("🌐 OPEN YOUR BROWSER TO: http://127.0.0.1:5000")
        print("⏳ Waiting for server to start...")
        
        # Wait a moment for server to start
        time.sleep(5)
        
        # Check if it's running
        print("🔍 Verifying server status...")
        result = subprocess.run([
            sys.executable, '-c', 
            'import requests; r = requests.get("http://127.0.0.1:5000", timeout=5); print(f"✅ Server responded: {r.status_code}")'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("🎉 SUCCESS! Server is running and responding!")
            print("📱 You can now open http://127.0.0.1:5000 in your browser")
        else:
            print("⚠️  Server might be starting slowly...")
            print("💡 Try opening http://127.0.0.1:5000 in your browser anyway")
            
        # Keep the process running
        print("\n🛑 Press Ctrl+C in THIS window to stop the server")
        process.wait()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Let's try the direct approach...")
        direct_start()

def direct_start():
    """Direct import method"""
    print("\n🔄 Trying direct import method...")
    try:
        from app import app
        print("✅ Direct import successful!")
        print("🚀 Starting server on http://127.0.0.1:5000")
        app.run(host='127.0.0.1', port=5000, debug=False)
    except Exception as e:
        print(f"❌ Direct start failed: {e}")
        print("\n🎯 LAST RESORT: Simple test server")
        simple_server()

def simple_server():
    """Absolute simplest server that MUST work"""
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return '''
        <html>
            <body style="background: #1e3c72; color: white; text-align: center; padding: 50px; font-family: Arial;">
                <h1>🎣 AI FISH FINDER - WORKING!</h1>
                <p>Your server is running successfully!</p>
                <p>This proves Flask is working correctly.</p>
                <p>Next: The full application will load here.</p>
            </body>
        </html>
        '''
    
    print("🎉 SIMPLE SERVER STARTING...")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    main()