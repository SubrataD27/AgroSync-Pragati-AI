import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import json

class DiseaseDetectionEngine:
    def __init__(self):
        """Initialize the disease detection engine."""
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')
        self.model_path = os.path.join(self.model_dir, 'plant_disease_model.h5')
        self.labels_path = os.path.join(self.model_dir, 'disease_labels.json')
        
        # Load the disease detection model
        self.model = self._load_model()
        self.labels = self._load_labels()
        
        # Disease information database
        self.disease_info = self._load_disease_info()
    
    def _load_model(self):
        """Load the trained disease detection model."""
        try:
            if os.path.exists(self.model_path):
                return load_model(self.model_path)
            else:
                print(f"Model not found at {self.model_path}. Disease detection unavailable.")
                return None
        except Exception as e:
            print(f"Error loading disease detection model: {e}")
            return None
    
    def _load_labels(self):
        """Load disease classification labels."""
        try:
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Labels not found at {self.labels_path}. Using default labels.")
                return {
                    str(i): f"disease_{i}" for i in range(38)  # Default for common plant disease dataset
                }
        except Exception as e:
            print(f"Error loading disease labels: {e}")
            return {}
    
    def _load_disease_info(self):
        """Load disease information database."""
        # This would typically come from a database, but we'll hardcode some examples
        return {
            "Apple___Apple_scab": {
                "name": "Apple Scab",
                "symptoms": "Olive green to brown or black spots on leaves and fruit. Infected leaves may twist, curl, or fall off.",
                "causes": "Fungus Venturia inaequalis, favored by cool, wet conditions during spring.",
                "treatment": [
                    "Apply fungicides early in the growing season",
                    "Remove and destroy fallen leaves",
                    "Prune trees to improve air circulation",
                    "Choose resistant apple varieties when planting"
                ],
                "prevention": [
                    "Plant resistant varieties",
                    "Apply preventative fungicide sprays",
                    "Maintain good sanitation by removing fallen leaves and fruit",
                    "Ensure adequate spacing between trees for airflow"
                ]
            },
            "Tomato___Early_blight": {
                "name": "Tomato Early Blight",
                "symptoms": "Dark brown spots with concentric rings on older leaves. Leaf yellowing around spots, eventual leaf drop.",
                "causes": "Fungus Alternaria solani, spread by wind, water, and contact. Thrives in warm, humid conditions.",
                "treatment": [
                    "Remove and destroy infected leaves",
                    "Apply appropriate fungicides",
                    "Maintain adequate plant spacing",
                    "Avoid overhead watering"
                ],
                "prevention": [
                    "Rotate crops every 2-3 years",
                    "Use disease-free seeds and transplants",
                    "Mulch around plants",
                    "Water at the base of plants",
                    "Stake plants for better air circulation"
                ]
            },
            "Corn___Common_rust": {
                "name": "Corn Common Rust",
                "symptoms": "Small, circular to elongated brown pustules on both leaf surfaces. Severe infections cause leaf yellowing and death.",
                "causes": "Fungus Puccinia sorghi, spreads via airborne spores.",
                "treatment": [
                    "Apply fungicides when rust is first observed",
                    "Remove heavily infected leaves if practical"
                ],
                "prevention": [
                    "Plant resistant corn varieties",
                    "Early planting can help avoid peak rust periods",
                    "Maintain good field sanitation",
                    "Monitor fields regularly for early detection"
                ]
            },
            "Healthy": {
                "name": "Healthy Plant",
                "symptoms": "No visible symptoms of disease.",
                "causes": "N/A",
                "treatment": ["No treatment necessary."],
                "prevention": [
                    "Continue good agricultural practices",
                    "Regular monitoring",
                    "Proper watering and fertilization",
                    "Integrated pest management"
                ]
            }
        }
    
    def preprocess_image(self, img_path):
        """
        Preprocess image for model input.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return preprocess_input(img_array)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def detect_disease(self, img_path):
        """
        Detect disease from plant image.
        
        Args:
            img_path (str): Path to the plant image
            
        Returns:
            dict: Detection results with disease info and confidence
        """
        if self.model is None:
            return {"error": "Disease detection model not loaded"}
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(img_path)
            if processed_img is None:
                return {"error": "Failed to process image"}
            
            # Make prediction
            predictions = self.model.predict(processed_img)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get label
            predicted_label = self.labels.get(str(predicted_class), f"Unknown disease ({predicted_class})")
            
            # Get disease information
            disease_details = self.disease_info.get(predicted_label, {
                "name": predicted_label,
                "symptoms": "Information not available",
                "causes": "Information not available",
                "treatment": ["Information not available"],
                "prevention": ["Information not available"]
            })
            
            # Create heatmap visualization
            heatmap_path = self._create_visualization(img_path, processed_img)
            
            return {
                "status": "success",
                "disease": predicted_label,
                "disease_name": disease_details["name"],
                "confidence": round(confidence * 100, 2),
                "symptoms": disease_details["symptoms"],
                "causes": disease_details["causes"],
                "treatment": disease_details["treatment"],
                "prevention": disease_details["prevention"],
                "visualization": heatmap_path if heatmap_path else None
            }
            
        except Exception as e:
            return {"error": f"Disease detection failed: {str(e)}"}
    
    def _create_visualization(self, original_img_path, processed_img):
        """
        Create visualization highlighting the affected areas.
        
        Args:
            original_img_path (str): Path to original image
            processed_img (numpy.ndarray): Preprocessed image array
            
        Returns:
            str: Path to visualization image
        """
        try:
            # This would typically use Grad-CAM or similar techniques
            # For simplicity, we'll just create a placeholder
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'assets', 'visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            # Read original image and resize
            img = cv2.imread(original_img_path)
            img = cv2.resize(img, (224, 224))
            
            # Create simple heatmap overlay (in a real app, use Grad-CAM or similar)
            # This is just a placeholder
            heatmap = np.zeros((224, 224), dtype=np.uint8)
            cv2.rectangle(heatmap, (50, 50), (150, 150), 255, -1)
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            
            # Overlay heatmap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
            
            # Save visualization
            base_filename = os.path.basename(original_img_path)
            output_path = os.path.join(output_dir, f"viz_{base_filename}")
            cv2.imwrite(output_path, superimposed)
            
            return output_path
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def update_disease_info(self, disease_id, info):
        """
        Update disease information in the database.
        
        Args:
            disease_id (str): Disease identifier
            info (dict): New disease information
            
        Returns:
            bool: Success status
        """
        try:
            if disease_id in self.disease_info:
                self.disease_info[disease_id].update(info)
                # In a real app, this would save to a database
                return True
            return False
        except Exception as e:
            print(f"Error updating disease info: {e}")
            return False