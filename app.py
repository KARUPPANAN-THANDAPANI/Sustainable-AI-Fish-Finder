from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import ee
from config import REGION_COORDINATES
from satellite_integration import RealTimeSatelliteData  # Import our new class

# Initialize Flask app
app = Flask(__name__)

# Load the trained AI model
try:
    model = joblib.load('fish_prediction_model.pkl')
    print("✅ AI Model loaded successfully")
except:
    print("❌ Could not load AI model")
    model = None

# Initialize satellite data engine
satellite_engine = RealTimeSatelliteData()

class PredictionEngine:
    def __init__(self):
        self.current_ocean_data = None
        self.update_ocean_data()
        
    def update_ocean_data(self):
        """Get fresh ocean data from satellites"""
        print("📡 Updating ocean data from satellites...")
        self.current_ocean_data = satellite_engine.get_daily_ocean_conditions()
        return self.current_ocean_data
    
    def predict_zone(self, lat, lon):
        """Predict fishing probability for a location using real ocean data"""
        if model is None:
            return {"error": "Model not loaded"}
        
        if self.current_ocean_data is None:
            self.update_ocean_data()
        
        # Use real ocean conditions + location
        sea_temp = self.current_ocean_data['sea_temperature']
        chlorophyll = self.current_ocean_data['chlorophyll']
        
        # Create feature array
        features = np.array([[sea_temp, chlorophyll, lat, lon]])
        
        try:
            # Get prediction probability
            probability = model.predict_proba(features)[0][1]
            
            # Determine zone type based on probability
            if probability > 0.7:
                zone_type = "high"
                color = "red"
                recommendation = "🎣 EXCELLENT! High fish activity expected"
            elif probability > 0.5:
                zone_type = "medium" 
                color = "yellow"
                recommendation = "🎣 GOOD fishing potential"
            elif probability > 0.3:
                zone_type = "low"
                color = "green" 
                recommendation = "🎣 Fair conditions - worth trying"
            else:
                zone_type = "very_low"
                color = "blue"
                recommendation = "🎣 Poor conditions - try other areas"
            
            return {
                'probability': round(float(probability) * 100, 1),
                'zone_type': zone_type,
                'color': color,
                'recommendation': recommendation,
                'confidence': f"{probability*100:.1f}%",
                'ocean_conditions': {
                    'temperature': sea_temp,
                    'chlorophyll': chlorophyll,
                    'data_source': self.current_ocean_data['data_source'],
                    'timestamp': self.current_ocean_data['timestamp']
                }
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize prediction engine
prediction_engine = PredictionEngine()

# Routes
@app.route('/')
def home():
    """Main page with fishing zone map"""
    ocean_data = prediction_engine.current_ocean_data
    return render_template('index.html', ocean_data=ocean_data)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        
        prediction = prediction_engine.predict_zone(lat, lon)
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/zones')
def get_zones():
    """Generate fishing zones based on real satellite data"""
    print("🎯 Generating fishing zones from satellite data...")
    heatmap_data = satellite_engine.generate_fishing_heatmap()
    return jsonify(heatmap_data)

@app.route('/update_data')
def update_data():
    """Force update of satellite data"""
    new_data = prediction_engine.update_ocean_data()
    return jsonify({
        "status": "success",
        "message": "Ocean data updated from satellites",
        "data": new_data
    })

@app.route('/ocean_report')
def ocean_report():
    """Get detailed ocean conditions report"""
    return jsonify(prediction_engine.current_ocean_data)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Fish Finder Web Application...")
    print("📡 Connected to NASA Satellite Feeds")
    print("🌐 Open: http://localhost:5000 in your browser")
    print(f"🌊 Current Ocean: {prediction_engine.current_ocean_data['sea_temperature']}°C")
    app.run(debug=True, host='0.0.0.0', port=5000)