import requests
import time

def check_server():
    print("🔍 Checking if server is running...")
    
    try:
        response = requests.get('http://127.0.0.1:5000', timeout=2)
        print(f"✅ SERVER IS RUNNING! Status: {response.status_code}")
        return True
    except requests.ConnectionError:
        print("❌ Connection refused - server not running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_server()