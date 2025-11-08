import ee
from config import REGION_COORDINATES

# Initialize
ee.Initialize(project='sustainable-fishing')

def check_export_status():
    """Check if our satellite data export completed"""
    print("🔍 Checking data export status...")
    
    # Get task list
    tasks = ee.batch.Task.list()
    found_tasks = False
    
    for task in tasks:
        config = task.config
        if 'description' in config and 'FishFinder' in config['description']:
            found_tasks = True
            print(f"📊 Task: {config.get('description', 'Unknown')}")
            print(f"🔄 Status: {task.state}")
            print(f"📁 Destination: {config.get('driveFolder', 'Root')}")
            print("---")
    
    if not found_tasks:
        print("❌ No FishFinder export tasks found.")
        print("Let's download fresh satellite data...")
        download_sample_data()
        return False
    
    return True

def download_sample_data():
    """Download fresh satellite data for analysis"""
    print("🚀 Downloading new satellite data...")
    
    region = ee.Geometry.Rectangle(REGION_COORDINATES)
    
    # Use reliable MODIS data
    modis_data = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterBounds(region) \
        .filterDate('2023-06-01', '2023-06-10') \
        .select('LST_Day_1km')  # Land Surface Temperature
    
    count = modis_data.size().getInfo()
    print(f"📊 Found {count} satellite images")
    
    if count > 0:
        # Export first image
        task = ee.batch.Export.image.toDrive(
            image=modis_data.first(),
            description='FishFinder_Day2_Data',
            folder='FishFinder_Project',
            scale=1000,
            region=region,
            maxPixels=1e9,
            fileFormat='GeoTIFF'
        )
        task.start()
        print("📤 New export started: 'FishFinder_Day2_Data'")
        print("✅ This will take 5-15 minutes to complete.")
        print("🔄 Let's proceed with sample data for now.")
    else:
        print("❌ No satellite data available for these dates.")

# Check existing data
if check_export_status():
    print("\n✅ Export tasks found. Data is processing.")
    print("📝 We can work with sample data while waiting.")
else:
    print("\n🔄 Starting fresh data download process.")

print("\n🎯 Let's proceed to data analysis with available data!")