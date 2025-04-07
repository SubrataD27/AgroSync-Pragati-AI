import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
import joblib

class CropRecommendationEngine:
    def __init__(self, model_path=None):
        """Initialize the crop recommendation engine with pre-trained models."""
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')
        
        # Default model path if not provided
        if model_path is None:
            self.model_path = os.path.join(self.model_dir, 'crop_recommendation_model.pkl')
        else:
            self.model_path = model_path
            
        # Load model if exists, otherwise prepare for training
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        
        # Define crop suitability thresholds
        self.suitability_thresholds = {
            'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temperature': (22, 30), 'humidity': (80, 90), 'ph': (5.5, 6.5), 'rainfall': (200, 300)},
            'wheat': {'N': (100, 140), 'P': (50, 70), 'K': (40, 70), 'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.0), 'rainfall': (75, 150)},
            'maize': {'N': (120, 160), 'P': (60, 80), 'K': (80, 100), 'temperature': (20, 30), 'humidity': (50, 80), 'ph': (5.5, 7.0), 'rainfall': (80, 200)},
            'chickpea': {'N': (40, 80), 'P': (60, 90), 'K': (30, 50), 'temperature': (20, 28), 'humidity': (40, 60), 'ph': (6.0, 8.0), 'rainfall': (60, 100)},
            'kidneybeans': {'N': (60, 100), 'P': (50, 80), 'K': (30, 60), 'temperature': (18, 28), 'humidity': (50, 70), 'ph': (5.5, 6.5), 'rainfall': (90, 140)},
            'pigeonpeas': {'N': (40, 80), 'P': (30, 60), 'K': (20, 40), 'temperature': (20, 32), 'humidity': (50, 80), 'ph': (5.5, 7.0), 'rainfall': (70, 120)},
            'mothbeans': {'N': (30, 60), 'P': (30, 50), 'K': (20, 40), 'temperature': (25, 35), 'humidity': (30, 60), 'ph': (6.0, 7.5), 'rainfall': (40, 100)},
            'mungbean': {'N': (40, 80), 'P': (40, 60), 'K': (20, 40), 'temperature': (25, 35), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (60, 120)},
            'blackgram': {'N': (40, 80), 'P': (30, 60), 'K': (20, 40), 'temperature': (25, 35), 'humidity': (60, 80), 'ph': (5.5, 7.0), 'rainfall': (70, 120)},
            'lentil': {'N': (30, 70), 'P': (40, 70), 'K': (20, 50), 'temperature': (18, 30), 'humidity': (40, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 110)},
            'pomegranate': {'N': (60, 100), 'P': (50, 80), 'K': (40, 80), 'temperature': (25, 35), 'humidity': (40, 60), 'ph': (6.0, 8.0), 'rainfall': (50, 100)},
            'banana': {'N': (100, 140), 'P': (70, 100), 'K': (120, 180), 'temperature': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (120, 200)},
            'mango': {'N': (80, 120), 'P': (50, 75), 'K': (80, 120), 'temperature': (24, 35), 'humidity': (50, 80), 'ph': (5.5, 7.5), 'rainfall': (100, 150)},
            'grapes': {'N': (70, 110), 'P': (60, 90), 'K': (100, 140), 'temperature': (20, 30), 'humidity': (50, 80), 'ph': (6.0, 7.5), 'rainfall': (60, 100)},
            'watermelon': {'N': (80, 120), 'P': (50, 80), 'K': (80, 120), 'temperature': (24, 32), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (50, 90)},
            'muskmelon': {'N': (70, 110), 'P': (50, 80), 'K': (70, 110), 'temperature': (24, 32), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (40, 80)},
            'apple': {'N': (70, 100), 'P': (60, 90), 'K': (70, 100), 'temperature': (15, 24), 'humidity': (60, 80), 'ph': (5.5, 6.5), 'rainfall': (100, 150)},
            'orange': {'N': (80, 120), 'P': (40, 70), 'K': (60, 100), 'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.5, 7.0), 'rainfall': (90, 130)},
            'papaya': {'N': (100, 140), 'P': (50, 80), 'K': (120, 160), 'temperature': (22, 32), 'humidity': (60, 85), 'ph': (6.0, 7.0), 'rainfall': (100, 180)},
            'coconut': {'N': (80, 120), 'P': (50, 80), 'K': (100, 150), 'temperature': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 250)},
            'cotton': {'N': (100, 140), 'P': (40, 70), 'K': (40, 70), 'temperature': (22, 32), 'humidity': (50, 80), 'ph': (6.0, 8.0), 'rainfall': (80, 120)},
            'jute': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temperature': (24, 35), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (150, 250)},
            'coffee': {'N': (100, 150), 'P': (40, 80), 'K': (80, 120), 'temperature': (18, 28), 'humidity': (70, 90), 'ph': (5.5, 6.5), 'rainfall': (150, 250)}
        }
        
        # Add crop growing seasons and regional suitability
        self.crop_seasons = {
            'rice': ['Kharif', 'Rabi'],
            'wheat': ['Rabi'],
            'maize': ['Kharif', 'Rabi'],
            'chickpea': ['Rabi'],
            'kidneybeans': ['Kharif'],
            'pigeonpeas': ['Kharif'],
            'mothbeans': ['Kharif'],
            'mungbean': ['Kharif'],
            'blackgram': ['Kharif'],
            'lentil': ['Rabi'],
            'pomegranate': ['Perennial'],
            'banana': ['Perennial'],
            'mango': ['Perennial'],
            'grapes': ['Perennial'],
            'watermelon': ['Summer'],
            'muskmelon': ['Summer'],
            'apple': ['Perennial'],
            'orange': ['Perennial'],
            'papaya': ['Perennial'],
            'coconut': ['Perennial'],
            'cotton': ['Kharif'],
            'jute': ['Kharif'],
            'coffee': ['Perennial']
        }
        
    def _load_model(self):
        """Load the pre-trained model if available."""
        try:
            if os.path.exists(self.model_path):
                return joblib.load(self.model_path)
            else:
                print(f"Model not found at {self.model_path}. Will need to train first.")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    def _load_scaler(self):
        """Load the data scaler if available."""
        scaler_path = os.path.join(self.model_dir, 'crop_scaler.pkl')
        try:
            if os.path.exists(scaler_path):
                return joblib.load(scaler_path)
            else:
                return StandardScaler()
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return StandardScaler()
    
    def get_recommendation(self, soil_data, climate_data, location=None, season=None):
        """
        Get crop recommendations based on soil, climate, and location data.
        
        Args:
            soil_data (dict): Dictionary containing soil parameters (N, P, K, pH)
            climate_data (dict): Dictionary containing climate parameters (temperature, humidity, rainfall)
            location (str, optional): Geographic location for regional recommendations
            season (str, optional): Current or planned growing season
            
        Returns:
            dict: Recommended crops with suitability scores and rationale
        """
        if self.model is None:
            return {"error": "Model not loaded. Please train the model first."}
        
        # Combine data
        input_data = {
            'N': soil_data.get('N', 0),
            'P': soil_data.get('P', 0),
            'K': soil_data.get('K', 0),
            'temperature': climate_data.get('temperature', 0),
            'humidity': climate_data.get('humidity', 0),
            'ph': soil_data.get('ph', 0),
            'rainfall': climate_data.get('rainfall', 0)
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        
        # Get model prediction
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Get top 5 predictions
        pred_indices = np.argsort(probabilities)[::-1][:5]
        class_names = self.model.classes_
        top_crops = [class_names[i] for i in pred_indices]
        top_probabilities = [probabilities[i] for i in pred_indices]
        
        # Prepare recommendations with explanations
        recommendations = []
        for crop, prob in zip(top_crops, top_probabilities):
            suitability = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
            
            # Check season suitability if season is provided
            season_match = True
            season_note = ""
            if season and crop in self.crop_seasons:
                if season not in self.crop_seasons[crop]:
                    season_match = False
                    season_note = f"{crop} is typically grown in {', '.join(self.crop_seasons[crop])} seasons, not in {season}."
            
            # Generate parameter analysis
            parameter_analysis = []
            thresholds = self.suitability_thresholds.get(crop, {})
            
            for param, value in input_data.items():
                if param in thresholds:
                    min_val, max_val = thresholds[param]
                    if min_val <= value <= max_val:
                        status = "optimal"
                    elif value < min_val:
                        status = "low"
                    else:
                        status = "high"
                    
                    parameter_analysis.append({
                        "parameter": param,
                        "value": value,
                        "optimal_range": f"{min_val}-{max_val}",
                        "status": status
                    })
            
            recommendations.append({
                "crop": crop,
                "confidence": round(prob * 100, 2),
                "suitability": suitability,
                "season_suitable": season_match,
                "season_note": season_note,
                "parameter_analysis": parameter_analysis
            })
        
        return {
            "primary_recommendation": recommendations[0]["crop"],
            "all_recommendations": recommendations,
            "input_parameters": input_data
        }
    
    def get_soil_improvement_tips(self, soil_data, recommended_crop):
        """
        Get tips to improve soil for the recommended crop.
        
        Args:
            soil_data (dict): Current soil parameters
            recommended_crop (str): The recommended crop
            
        Returns:
            list: Tips for soil improvement
        """
        tips = []
        if recommended_crop not in self.suitability_thresholds:
            return ["No specific soil improvement tips available for this crop."]
        
        thresholds = self.suitability_thresholds[recommended_crop]
        
        # Check nitrogen levels
        n_min, n_max = thresholds.get('N', (0, 0))
        if soil_data.get('N', 0) < n_min:
            tips.append("Increase nitrogen by adding nitrogen-rich fertilizers or organic matter like compost or manure.")
        elif soil_data.get('N', 0) > n_max:
            tips.append("Reduce nitrogen by planting nitrogen-fixing cover crops or adding carbon-rich materials.")
        
        # Check phosphorus levels
        p_min, p_max = thresholds.get('P', (0, 0))
        if soil_data.get('P', 0) < p_min:
            tips.append("Increase phosphorus by adding bone meal, rock phosphate, or phosphorus-specific fertilizers.")
        elif soil_data.get('P', 0) > p_max:
            tips.append("Reduce phosphorus applications and consider soil testing regularly.")
        
        # Check potassium levels
        k_min, k_max = thresholds.get('K', (0, 0))
        if soil_data.get('K', 0) < k_min:
            tips.append("Increase potassium by adding wood ash, seaweed, or potassium-specific fertilizers.")
        elif soil_data.get('K', 0) > k_max:
            tips.append("Avoid adding more potassium fertilizers and focus on balancing nutrients.")
        
        # Check pH levels
        ph_min, ph_max = thresholds.get('ph', (0, 0))
        if soil_data.get('ph', 0) < ph_min:
            tips.append(f"Increase soil pH (currently acidic) by adding agricultural lime to reach optimal range of {ph_min}-{ph_max}.")
        elif soil_data.get('ph', 0) > ph_max:
            tips.append(f"Decrease soil pH (currently alkaline) by adding sulfur, peat moss, or aluminum sulfate to reach optimal range of {ph_min}-{ph_max}.")
        
        if not tips:
            tips.append("Your soil conditions are already well-suited for this crop.")
            
        return tips
    
    def train_model(self, training_data_path):
        """
        Train the crop recommendation model with new data.
        
        Args:
            training_data_path (str): Path to the training data CSV
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Load training data
            df = pd.read_csv(training_data_path)
            
            # Prepare features and target
            X = df.drop('label', axis=1)
            y = df['label']
            
            # Scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'crop_scaler.pkl'))
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False