import ee

# Test Earth Engine connection
try:
    ee.Initialize()
    print("✅ Earth Engine connected successfully!")
    
    # Test region
    region = ee.Geometry.Rectangle([80, 5, 90, 15])
    print("✅ Region defined:", region.getInfo())
    
except Exception as e:
    print("❌ Need to authenticate first:")
    print("Run: ee.Authenticate() in terminal")