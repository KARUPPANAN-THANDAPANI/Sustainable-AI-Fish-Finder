import ee

# Method 1: Initialize with project (use the project name from your Earth Engine setup)
try:
    ee.Initialize(project='sustainable-fishing')
    print("✅ Earth Engine initialized with project 'sustainable-fishing'!")
except:
    print("❌ Project 'sustainable-fishing' not found")

# Method 2: Initialize without specific project (let it use default)
try:
    ee.Initialize()
    print("✅ Earth Engine initialized with default project!")
except Exception as e:
    print(f"❌ Default initialization failed: {e}")
    print("\nLet's authenticate properly...")
    
    # Re-authenticate
    ee.Authenticate()
    ee.Initialize()
    print("✅ Re-authenticated and initialized!")