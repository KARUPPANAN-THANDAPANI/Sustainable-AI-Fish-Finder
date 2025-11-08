import ee
import time

print("Testing Earth Engine connection...")

try:
    # Initialize
    ee.Initialize()
    print("✅ Earth Engine initialized!")
    
    # Small test
    point = ee.Geometry.Point([80, 10])
    print("✅ Geometry test passed!")
    
    # Test with a simple image
    image = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    print("✅ Can access satellite data!")
    
    print("\n🎉 ALL TESTS PASSED! You're ready to build your fish finder!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Try running: ee.Authenticate() again")