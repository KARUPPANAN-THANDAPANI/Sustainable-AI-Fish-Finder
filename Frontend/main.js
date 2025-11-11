// Step 1: Initialize Leaflet map
const map = L.map('map').setView([10.5, 79.1], 6); // Tamil Nadu coast

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors'
}).addTo(map);
const regionCoordinates = {
  "chennai": [13.0827, 80.2707],
  "tuticorin": [8.7642, 78.1348],
  "rameswaram": [9.2876, 79.3129],
  "puducherry": [11.9416, 79.8083],
  "cuddalore": [11.7500, 79.7500],
  "nagapattinam": [10.7667, 79.8500],
  "kanyakumari": [8.0883, 77.5385],
  "chidambaram": [11.4000, 79.7000],
  "mahabalipuram": [12.6131, 80.1950]
};
  
// Step 2 & 3: Handle form submission and fetch prediction
document.getElementById('input-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const region = document.getElementById('region').value;
  const date = document.getElementById('date').value;

  fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    body: JSON.stringify({ region, date }),
    headers: { 'Content-Type': 'application/json' }
  })
  .then(res => res.json())
  .then(data => {
    console.log("Prediction received:", data);
    displayPrediction(data);      // Step 5: Text summary
    loadPredictionZones();        // Step 4: GeoJSON overlay
  })
  .catch(error => {
    console.error("Error fetching prediction:", error);
  });
});

// Step 5: Display prediction summary
function displayPrediction(data) {
  const summaryDiv = document.getElementById('summary');
  summaryDiv.innerHTML = `
    <h3>Prediction Summary</h3>
    <p><strong>Region:</strong> ${data.region}</p>
    <p><strong>Date:</strong> ${data.date}</p>
    <p><strong>Prediction:</strong> ${data.prediction}</p>
  `;
  const coords = regionCoordinates[data.region] || [10.5, 79.1];
  map.setView(coords, 8);
  const marker = L.marker(coords).addTo(map);
  marker.bindPopup(`Prediction: ${data.prediction}`).openPopup();
}

// Step 4: Load GeoJSON overlay
function loadPredictionZones() {
  const region = document.getElementById('region').value;
  fetch(`http://127.0.0.1:5000/zones?region=${region}`)
    .then(res => res.json())
    .then(geojson => {
      L.geoJSON(geojson, {
        style: feature => ({
          color: feature.properties.risk === 'high' ? 'green' : 'blue',
          weight: 2,
          fillOpacity: 0.5
        }),
        onEachFeature: (feature, layer) => {
          layer.bindPopup(`Zone: ${feature.properties.name}<br>Risk: ${feature.properties.risk}`);
        }
      }).addTo(map);
    });

  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'info legend');
    div.innerHTML = `
      <strong>Fish Likelihood</strong><br>
      <i style="background: green; width: 12px; height: 12px; display: inline-block;"></i> High<br>
      <i style="background: blue; width: 12px; height: 12px; display: inline-block;"></i> Moderate<br>
    `;
    return div;
  };
  legend.addTo(map);
}

// Step 5: Export map as PNG
const printer = L.easyPrint({
  tileLayer: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'),
  sizeModes: ['Current'],
  filename: 'fish_prediction_map',
  exportOnly: true,
  hideControlContainer: true
}).addTo(map);

document.getElementById('export-map').addEventListener('click', () => {
  printer.printMap('CurrentSize', 'fish_prediction_map');
});

// Step 5: Export data as CSV
document.getElementById('export-csv').addEventListener('click', () => {
  const region = document.getElementById('region').value;
  const date = document.getElementById('date').value;
  const prediction = document.querySelector('#summary p:nth-child(4)').textContent.split(': ')[1];

  const csvContent = `Region,Date,Prediction\n${region},${date},${prediction}`;
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'fish_prediction.csv';
  link.click();
});
// Step 6: Submit feedback
document.getElementById('feedback-form').addEventListener('submit', function(e) {
  e.preventDefault();
  const useful = document.querySelector('input[name="useful"]:checked').value;
  const comments = document.getElementById('comments').value;
  const region = document.getElementById('region').value;
  const date = document.getElementById('date').value;
  fetch('http://127.0.0.1:5000/feedback',{
    method: 'POST',
    body: JSON.stringify({ region, date, useful, comments }),
    headers: { 'Content-Type': 'application/json' }
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('feedback-response').innerText = "Thanks for your feedback!";
    document.getElementById('feedback-form').reset();
  })
  .catch(error => {
    console.error("Error submitting feedback:", error);
    document.getElementById('feedback-response').innerText = "Error submitting feedback.";
  });
});
  