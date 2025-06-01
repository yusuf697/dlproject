import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("artifacts", "training", "model.h5"))
        
        # ✅ Class mapping — change this only if your model uses different label order
        self.class_indices = {0: 'Normal', 1: 'Tumor'}  # <-- Flip if predictions are reversed

    def predict(self):
        try:
            # Load and preprocess image
            img = image.load_img(self.filename, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            probs = self.model.predict(img_array)
            predicted_class = np.argmax(probs, axis=1)[0]
            prediction_label = self.class_indices.get(predicted_class, 'Unknown')
            confidence = float(probs[0][predicted_class])

            # Log for debugging
            print(f"[INFO] Predicted Class: {predicted_class} | Label: {prediction_label} | Confidence: {confidence:.2f}")

            return [{
                "image": prediction_label,
                "confidence": f"{confidence * 100:.2f}%"
            }]
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return [{"error": str(e)}]
