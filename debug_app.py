from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import ee
import traceback
from config import REGION_COORDINATES

app = Flask(__name__)

# Initialize with error handling
try:
    model = joblib.load('fish_prediction_model.pkl')
    print("✅ AI Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

class PredictionEngine:
    def __init__(self):
        try:
            ee.Initialize(project='sustainable-fishing')
            self.region = ee.Geometry.Rectangle(REGION_COORDINATES)
            self.current_ocean_data = self.get_fallback_data()
            print("✅ Prediction Engine initialized")
        except Exception as e:
            print(f"❌ Earth Engine init error: {e}")
            self.current_ocean_data = self.get_fallback_data()
    
    def get_fallback_data(self):
        """Provide fallback data"""
        return {
            'sea_temperature': 28.5,
            'chlorophyll': 0.4,
            'current_speed': 0.5,
            'data_source': 'Fallback Data',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'region_name': 'Bay of Bengal'
        }
    
    def predict_zone(self, lat, lon):
        """Predict fishing probability"""
        if model is None:
            return {"error": "Model not loaded"}
        
        try:
            sea_temp = self.current_ocean_data['sea_temperature']
            chlorophyll = self.current_ocean_data['chlorophyll']
            
            features = np.array([[sea_temp, chlorophyll, lat, lon]])
            probability = model.predict_proba(features)[0][1]
            
            if probability > 0.7:
                zone_type = "high"
                recommendation = "🎣 EXCELLENT! High fish activity expected"
            elif probability > 0.5:
                zone_type = "medium" 
                recommendation = "🎣 GOOD fishing potential"
            else:
                zone_type = "low"
                recommendation = "🎣 Fair conditions - worth trying"
            
            return {
                'probability': round(float(probability) * 100, 1),
                'zone_type': zone_type,
                'recommendation': recommendation,
                'confidence': f"{probability*100:.1f}%"
            }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

# Initialize engine
prediction_engine = PredictionEngine()

@app.route('/')
def home():
    try:
        ocean_data = prediction_engine.current_ocean_data
        return render_template('index.html', ocean_data=ocean_data)
    except Exception as e:
        return f"""
        <html>
            <body style="background: #1e3c72; color: white; padding: 50px; text-align: center;">
                <h1>🎣 AI Fish Finder</h1>
                <h2>Template Error</h2>
                <p>Error: {str(e)}</p>
                <p>Check that templates/index.html exists</p>
            </body>
        </html>
        """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lat = float(data['latitude'])
        lon = float(data['longitude'])
        prediction = prediction_engine.predict_zone(lat, lon)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": f"Prediction endpoint error: {str(e)}"})

@app.route('/zones')
def get_zones():
    """Generate sample fishing zones"""
    try:
        zones = []
        # Create sample zones
        for i in range(5):
            zones.append({
                'lat': np.random.uniform(8, 12),
                'lon': np.random.uniform(82, 88), 
                'probability': np.random.uniform(70, 90),
                'type': 'high',
                'color': 'red'
            })
        for i in range(8):
            zones.append({
                'lat': np.random.uniform(5, 15),
                'lon': np.random.uniform(80, 90),
                'probability': np.random.uniform(40, 69),
                'type': 'medium',
                'color': 'yellow'
            })
        return jsonify(zones)
    except Exception as e:
        return jsonify({"error": f"Zones error: {str(e)}"})

@app.route('/update_data')
def update_data():
    return jsonify({
        "status": "success",
        "message": "Data update simulated",
        "data": prediction_engine.current_ocean_data
    })

if __name__ == '__main__':
    print("🚀 Starting DEBUG Fish Finder...")
    print("🌐 Open: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)