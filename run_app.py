from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow frontend requests from browser

# Step 3: Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    region = data['region']
    date = data['date']
    # Placeholder logic — replace with model call
    result = {"prediction": "High fish density", "region": region, "date": date}
    return jsonify(result)

# Step 4: Region-specific GeoJSON zone overlay
@app.route('/zones')
def zones():
    region = request.args.get('region', '').lower()
    geojson_folder = os.path.json(os.getcwd(), 'geojson')
    region_file_map = {
        'chennai': 'chennai_zones.geojson',
        'tuticorin': 'tuticorin_zones.geojson',
        'rameswaram': 'rameswaram_zones.geojson',
        'nagapattinam': 'nagapattinam_zones.geojson',
        'puducherry': 'puducherry_zones.geojson',
        'cuddalore': 'cuddalore_zones.geojson',
        'kanyakumari': 'kanyakumari_zones.geojson',
        'chidambaram': 'chidambaram_zones.geojson',
        'mahabalipuram': 'mahabalipuram_zones.geojson'
    }
    filename = region_file_map.get(region)
    if filename:
        filepath = os.path.join(geojson_folder, filename)
        if os.path.exists(filepath):
            return send_file(filepath)
    return jsonify({"type": "FeatureCollection", "features": []})
    
# Step 5: Feedback endpoint
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    region = data['region']
    date = data['date']
    useful = data['useful']
    comments = data.get('comments')
    print(f"Feedback received: Region={region}, Date={date}, Useful={useful}, Comments={comments}")
    with open('feedback.log.csv', 'a') as f:
        f.write(f"{region},{date},{useful},{comments}\n")
    return jsonify({"status": "success","message": "Feedback received"})
if __name__ == '__main__':
    app.run(debug=True)