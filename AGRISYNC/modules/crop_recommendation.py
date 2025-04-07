import pandas as pd
import numpy as np
import joblib
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

class CropRecommendationSystem:
    """Crop recommendation system based on soil and environmental parameters"""
    
    def __init__(self):
        self.model = None
        self.crops = [
            'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
            'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'
        ]
        self.load_model()
        
    def load_model(self):
        """Load the crop recommendation model"""
        if self.model is None:
            try:
                model_path = os.path.join("models", "crop_recommendation_model.pkl")
                
                # If model doesn't exist, train a new one with sample data
                if not os.path.exists(model_path):
                    st.info("Training crop recommendation model...")
                    self.train_model()
                else:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
            except Exception as e:
                st.error(f"Error loading crop recommendation model: {e}")
                # Train a fallback model
                self.train_model()
    
    def train_model(self):
        """Train a new crop recommendation model with sample data"""
        try:
            # Create the models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Generate synthetic training data
            df = self.generate_synthetic_data()
            
            # Prepare features and target
            X = df.drop('label', axis=1)
            y = df['label']
            
            # Train a Random Forest model
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X, y)
            
            # Save the model
            model_path = os.path.join("models", "crop_recommendation_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.model = model
            
        except Exception as e:
            st.error(f"Error training crop recommendation model: {e}")
    
    def generate_synthetic_data(self):
        """Generate synthetic crop data for training"""
        # Define typical ranges for each parameter
        param_ranges = {
            'N': (0, 140),     # Nitrogen (kg/ha)
            'P': (5, 145),     # Phosphorus (kg/ha)
            'K': (5, 205),     # Potassium (kg/ha)
            'temperature': (8, 44),  # Temperature (°C)
            'humidity': (14, 100),   # Humidity (%)
            'ph': (3.5, 10),   # pH
            'rainfall': (20, 300)    # Rainfall (mm)
        }
        
        # Define optimal parameters for each crop
        crop_params = {
            'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 
                    'temperature': (22, 28), 'humidity': (80, 95), 
                    'ph': (5.5, 6.5), 'rainfall': (200, 300)},
            'maize': {'N': (80, 120), 'P': (40, 60), 'K': (30, 50), 
                     'temperature': (21, 27), 'humidity': (50, 75), 
                     'ph': (5.5, 7.0), 'rainfall': (90, 140)},
            'chickpea': {'N': (40, 60), 'P': (60, 80), 'K': (20, 40), 
                        'temperature': (20, 25), 'humidity': (40, 60), 
                        'ph': (6.0, 8.0), 'rainfall': (60, 90)},
            'kidneybeans': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 
                          'temperature': (20, 25), 'humidity': (40, 60), 
                          'ph': (5.5, 7.0), 'rainfall': (90, 120)},
            'pigeonpeas': {'N': (20, 40), 'P': (60, 80), 'K': (20, 40), 
                         'temperature': (25, 30), 'humidity': (70, 80), 
                         'ph': (5.5, 7.0), 'rainfall': (60, 90)},
            'mothbeans': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 
                        'temperature': (25, 30), 'humidity': (30, 50), 
                        'ph': (5.0, 7.0), 'rainfall': (40, 70)},
            'mungbean': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 
                       'temperature': (25, 30), 'humidity': (70, 80), 
                       'ph': (6.0, 7.5), 'rainfall': (70, 100)},
            'blackgram': {'N': (40, 60), 'P': (40, 60), 'K': (20, 40), 
                        'temperature': (25, 30), 'humidity': (70, 80), 
                        'ph': (6.0, 7.5), 'rainfall': (70, 100)},
            'lentil': {'N': (20, 40), 'P': (40, 60), 'K': (20, 40), 
                     'temperature': (20, 25), 'humidity': (40, 60), 
                     'ph': (6.0, 7.0), 'rainfall': (50, 80)},
            'pomegranate': {'N': (20, 40), 'P': (20, 40), 'K': (40, 60), 
                          'temperature': (20, 30), 'humidity': (50, 60), 
                          'ph': (5.5, 7.5), 'rainfall': (50, 80)},
            'banana': {'N': (100, 140), 'P': (20, 40), 'K': (100, 140), 
                     'temperature': (25, 35), 'humidity': (70, 80), 
                     'ph': (6.0, 7.5), 'rainfall': (120, 180)},
            'mango': {'N': (20, 40), 'P': (20, 40), 'K': (30, 50), 
                    'temperature': (25, 35), 'humidity': (60, 80), 
                    'ph': (5.5, 7.5), 'rainfall': (100, 150)},
            'grapes': {'N': (20, 40), 'P': (20, 40), 'K': (40, 60), 
                     'temperature': (20, 30), 'humidity': (70, 80), 
                     'ph': (6.0, 7.5), 'rainfall': (80, 110)},
            'watermelon': {'N': (60, 80), 'P': (40, 60), 'K': (60, 80), 
                         'temperature': (25, 35), 'humidity': (60, 70), 
                         'ph': (6.0, 7.0), 'rainfall': (40, 70)},
            'muskmelon': {'N': (60, 80), 'P': (40, 60), 'K': (40, 60), 
                        'temperature': (25, 35), 'humidity': (60, 70), 
                        'ph': (6.0, 7.0), 'rainfall': (40, 70)},
            'apple': {'N': (20, 40), 'P': (20, 40), 'K': (20, 40), 
                    'temperature': (10, 20), 'humidity': (70, 80), 
                    'ph': (5.5, 6.5), 'rainfall': (100, 140)},
            'orange': {'N': (20, 40), 'P': (10, 30), 'K': (30, 50), 
                     'temperature': (20, 30), 'humidity': (70, 80), 
                     'ph': (5.5, 7.0), 'rainfall': (100, 150)},
            'papaya': {'N': (40, 60), 'P': (20, 40), 'K': (40, 60), 
                     'temperature': (25, 35), 'humidity': (70, 80), 
                     'ph': (6.0, 7.0), 'rainfall': (100, 150)},
            'coconut': {'N': (20, 40), 'P': (20, 40), 'K': (80, 120), 
                      'temperature': (25, 35), 'humidity': (70, 80), 
                      'ph': (5.5, 7.0), 'rainfall': (150, 250)},
            'cotton': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 
                     'temperature': (25, 35), 'humidity': (50, 70), 
                     'ph': (6.0, 8.0), 'rainfall': (80, 110)},
            'jute': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 
                   'temperature': (25, 35), 'humidity': (70, 90), 
                   'ph': (6.0, 7.5), 'rainfall': (150, 200)},
            'coffee': {'N': (80, 120), 'P': (20, 40), 'K': (20, 40), 
                     'temperature': (20, 25), 'humidity': (70, 80), 
                     'ph': (6.0, 7.0), 'rainfall': (150, 200)}
        }
        
        # Generate synthetic data
        data = []
        for crop in self.crops:
            # Generate 50 samples per crop
            for _ in range(50):
                sample = {}
                
                # Generate values within the optimal range for this crop
                for param, (min_val, max_val) in param_ranges.items():
                    # Get crop-specific range
                    crop_min, crop_max = crop_params[crop].get(param, (min_val, max_val))
                    
                    # Generate value within crop-specific range (70% of the time)
                    if np.random.random() < 0.7:
                        value = np.random.uniform(crop_min, crop_max)
                    else:
                        # Generate value from general range
                        value = np.random.uniform(min_val, max_val)
                    
                    sample[param] = value
                
                sample['label'] = crop
                data.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    def recommend_crop(self, N, P, K, temperature, humidity, ph, rainfall):
        """Recommend crops based on soil and environmental parameters"""
        if self.model is None:
            self.load_model()
            
        try:
            # Prepare input data
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # Get predictions (probabilities)
            probabilities = self.model.predict_proba(input_data)[0]
            
            # Sort crops by probability
            crop_probabilities = [(crop, prob) for crop, prob in zip(self.model.classes_, probabilities)]
            crop_probabilities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5 recommendations with probabilities
            return crop_probabilities[:5]
        
        except Exception as e:
            st.error(f"Error recommending crop: {e}")
            return []
    
    def get_crop_info(self, crop_name):
        """Get information about a crop"""
        crop_info = {
            'rice': {
                'scientific_name': 'Oryza sativa',
                'growing_season': 'Kharif (June-July to October-November)',
                'water_requirements': 'High (1000-1500mm)',
                'fertilizer_requirements': 'N:P:K = 120:60:60 kg/ha',
                'disease_resistance': 'Moderate',
                'market_value': '₹1800-2200 per quintal',
                'description': 'Rice is one of the most important food crops in India, particularly in eastern and southern regions. It requires standing water for optimal growth.'
            },
            'maize': {
                'scientific_name': 'Zea mays',
                'growing