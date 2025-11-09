# model_validator.py
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import SatelliteDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self):
        self.processor = SatelliteDataProcessor()
        logger.info("🔍 Model Validator Initialized")
    
    def create_validation_dataset(self, num_samples=200):
        """Create a separate validation dataset"""
        logger.info("📊 Creating validation dataset...")
        
        validation_data = []
        np.random.seed(123)  # Different seed from training
        
        for i in range(num_samples):
            lat = np.random.uniform(5, 15)
            lon = np.random.uniform(80, 90)
            
            features = self.processor.extract_ocean_features(lat, lon)
            sst = features['sea_temperature']
            chloro = features['chlorophyll']
            
            # More realistic success probability based on oceanography
            success_prob = self.calculate_realistic_success(sst, chloro, lat, lon)
            fishing_success = 1 if np.random.random() < success_prob else 0
            
            validation_data.append({
                'sea_temperature': sst,
                'chlorophyll': chloro,
                'latitude': lat,
                'longitude': lon,
                'fishing_success': fishing_success,
                'true_probability': success_prob
            })
        
        df = pd.DataFrame(validation_data)
        df.to_csv('validation_dataset.csv', index=False)
        logger.info(f"✅ Validation dataset created: {len(df)} samples")
        return df
    
    def calculate_realistic_success(self, sst, chloro, lat, lon):
        """More realistic fishing success based on marine science"""
        success_prob = 0.3  # Base probability
        
        # SST optimization (species-specific)
        if 26 <= sst <= 30:  # Tropical fish optimal range
            success_prob += 0.3
        elif sst < 20 or sst > 32:  # Too cold/hot
            success_prob -= 0.2
        
        # Chlorophyll optimization (plankton blooms)
        if 0.2 <= chloro <= 0.8:  # Good productivity
            success_prob += 0.25
        elif chloro > 1.0:  # Too turbid
            success_prob -= 0.1
        
        # Geographic preferences (known fishing grounds)
        if 8 <= lat <= 12 and 82 <= lon <= 88:  # Prime fishing area
            success_prob += 0.15
        
        # Depth consideration (simplified)
        depth_factor = 1.0 - (abs(lon - 85) * 0.02)  # Deeper water preference
        success_prob *= depth_factor
        
        return max(0.05, min(0.95, success_prob))
    
    def validate_model_performance(self, validation_df):
        """Comprehensive model validation"""
        logger.info("🎯 Validating model performance...")
        
        # Load the trained model
        try:
            import joblib
            model = joblib.load('fish_prediction_model.pkl')
        except:
            logger.error("❌ No trained model found. Training new one...")
            training_data = self.processor.create_training_dataset(300)
            accuracy = self.processor.train_random_forest_model(training_data)
            model = self.processor.model
        
        # Prepare features
        features = ['sea_temperature', 'chlorophyll', 'latitude', 'longitude']
        X_val = validation_df[features]
        y_true = validation_df['fishing_success']
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        logger.info("📈 VALIDATION RESULTS:")
        logger.info(f"   Accuracy:  {accuracy:.3f}")
        logger.info(f"   Precision: {precision:.3f}")
        logger.info(f"   Recall:    {recall:.3f}")
        logger.info(f"   F1-Score:  {f1:.3f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"   Confusion Matrix:\n{cm}")
        
        # Feature performance analysis
        self.analyze_feature_performance(validation_df, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def analyze_feature_performance(self, df, predictions):
        """Analyze how well each feature correlates with predictions"""
        logger.info("🔍 Feature Performance Analysis:")
        
        correlations = {}
        for feature in ['sea_temperature', 'chlorophyll', 'latitude', 'longitude']:
            corr = np.corrcoef(df[feature], predictions)[0, 1]
            correlations[feature] = corr
            logger.info(f"   {feature}: {corr:.3f}")
        
        return correlations
    
    def test_specific_fishing_spots(self):
        """Test known fishing spots for validation"""
        logger.info("🎣 Testing known fishing spots...")
        
        # Known productive fishing spots in the region
        known_spots = [
            (10.2, 85.1, "Rameswaram Coast"),
            (9.3, 79.3, "Gulf of Mannar"),
            (8.1, 77.5, "Kanyakumari"),
            (13.1, 80.3, "Chennai Coast"),
            (15.5, 80.1, "Andhra Fishing Zone")
        ]
        
        results = []
        for lat, lon, name in known_spots:
            try:
                prediction = self.processor.predict_fishing_success(lat, lon)
                results.append({
                    'name': name,
                    'location': (lat, lon),
                    'predicted_success': prediction['probability'],
                    'prediction': 'Good' if prediction['probability'] > 0.5 else 'Poor'
                })
                logger.info(f"   {name}: {prediction['probability']:.1%}")
            except Exception as e:
                logger.error(f"   {name}: Error - {e}")
        
        return results

# Run comprehensive validation
if __name__ == "__main__":
    validator = ModelValidator()
    
    print("🚀 DAY 7-8: MODEL VALIDATION & REFINEMENT")
    print("=" * 50)
    
    # Step 1: Create validation dataset
    print("\n📊 STEP 1: Creating Validation Dataset")
    validation_data = validator.create_validation_dataset(150)
    
    # Step 2: Validate model performance
    print("\n🎯 STEP 2: Model Performance Validation")
    metrics = validator.validate_model_performance(validation_data)
    
    # Step 3: Test known fishing spots
    print("\n🎣 STEP 3: Testing Known Fishing Spots")
    spot_results = validator.test_specific_fishing_spots()
    
    # Step 4: Summary
    print("\n📋 VALIDATION SUMMARY:")
    print(f"   ✅ Model Accuracy: {metrics['accuracy']:.1%}")
    print(f"   ✅ F1-Score: {metrics['f1_score']:.3f}")
    print(f"   ✅ Known spots tested: {len(spot_results)}")
    
    # Calculate validation score
    validation_score = (metrics['accuracy'] + metrics['f1_score']) / 2
    print(f"   🎯 Overall Validation Score: {validation_score:.3f}")
    
    if validation_score >= 0.7:
        print("   🎉 Model validation PASSED - Ready for production!")
    else:
        print("   ⚠️  Model needs improvement - Consider retraining with more data")