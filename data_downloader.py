import ee
import pandas as pd
from config import REGION_COORDINATES, SATELLITE_SOURCES, DATA_START_DATE, DATA_END_DATE

# Initialize Earth Engine with YOUR project
ee.Initialize(project='sustainable-fishing')

def download_satellite_data():
    """Download SST and Chlorophyll data for fish prediction"""
    
    print("🚀 Starting satellite data download...")
    
    # Define fishing region
    region = ee.Geometry.Rectangle(REGION_COORDINATES)
    print(f"📍 Target region: {REGION_COORDINATES}")
    
    # Get MODIS Sea Surface Temperature data
    sst_data = ee.ImageCollection(SATELLITE_SOURCES['modis_sst']) \
        .filterBounds(region) \
        .filterDate(DATA_START_DATE, DATA_END_DATE) \
        .select('LST_Day_1km')  # Land Surface Temperature
    
    # Get data count
    data_count = sst_data.size().getInfo()
    print(f"📊 Found {data_count} satellite images")
    
    if data_count > 0:
        # Get first image info
        first_image = sst_data.first()
        print("✅ Successfully accessed satellite data!")
        print(f"📅 Date range: {DATA_START_DATE} to {DATA_END_DATE}")
        
        # Export sample data (this will take a few minutes)
        export_task = ee.batch.Export.image.toDrive(
            image=first_image,
            description='FishFinder_SST_Data',
            scale=1000,
            region=region,
            maxPixels=1e9
        )
        export_task.start()
        print("📤 Export task started! Check Google Drive for your data.")
        
        return True
    else:
        print("❌ No data found for the specified dates.")
        return False

if __name__ == "__main__":
    success = download_satellite_data()
    if success:
        print("\n🎉 PHASE 1 COMPLETE! You've successfully:")
        print("   ✅ Set up Earth Engine access")
        print("   ✅ Connected to NASA satellite data")
        print("   ✅ Started downloading ocean temperature data")
        print("   ✅ Built the foundation for your AI fish finder!")
    else:
        print("\n⚠️  Some issues encountered. Check your configuration.")