import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the path to your test images directory
TEST_IMAGES_DIR = 'C:\\Users\\akbar\\OneDrive\\Documents\\Kissancare\\processed_data_backup_20250522_160421\\test' # Use the absolute path

# Define the path to your trained model (replace with the actual path if different)
MODEL_PATH = r'C:\Users\akbar\OneDrive\Documents\Kissancare\logs_debug\20250528_141305\best_model.h5' # Use the absolute path to your best model file

# Define the image size your model expects
MODEL_IMAGE_SIZE = 224 # Replace with your actual model's image size

# Define the class mapping (replace with your actual mapping)
# This should match the CLASS_INDEX_TO_DISEASE dictionary in your app.py
CLASS_INDEX_TO_DISEASE = {
    '0': 'Apple___Apple_scab',
    '1': 'Apple___Black_rot',
    '2': 'Apple___Cedar_apple_rust',
    '3': 'Apple___healthy',
    '4': 'Blueberry___healthy',
    '5': 'Cherry_(including_sour)___Powdery_mildew',
    '6': 'Cherry_(including_sour)___healthy',
    '7': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    '8': 'Corn_(maize)___Common_rust_',
    '9': 'Corn_(maize)___Northern_Leaf_Blight',
    '10': 'Corn_(maize)___healthy',
    '11': 'Grape___Black_rot',
    '12': 'Grape___Esca_(Black_Measles)',
    '13': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    '14': 'Grape___healthy',
    '15': 'Orange___Haunglongbing_(Citrus_greening)',
    '16': 'Peach___Bacterial_spot',
    '17': 'Peach___healthy',
    '18': 'Pepper,_bell___Bacterial_spot',
    '19': 'Pepper,_bell___healthy',
    '20': 'Potato___Early_blight',
    '21': 'Potato___Late_blight',
    '22': 'Potato___healthy',
    '23': 'Raspberry___healthy',
    '24': 'Soybean___healthy',
    '25': 'Squash___Powdery_mildew',
    '26': 'Strawberry___Leaf_scorch',
    '27': 'Strawberry___healthy',
    '28': 'Tomato___Bacterial_spot',
    '29': 'Tomato___Early_blight',
    '30': 'Tomato___Late_blight',
    '31': 'Tomato___Leaf_Mold',
    '32': 'Tomato___Septoria_leaf_spot',
    '33': 'Tomato___Spider_mites Two-spotted_spider_mite',
    '34': 'Tomato___Target_Spot',
    '35': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    '36': 'Tomato___Tomato_mosaic_virus',
    '37': 'Tomato___healthy'
}

# Load the trained model
model = None
try:
    # Configure TensorFlow to be less verbose during loading
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = tf.keras.models.load_model(MODEL_PATH)
    os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Reset log level
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please check if the MODEL_PATH is correct and the file exists.")


if model:
    while True:
        image_path = input("Enter the full path to the image file (or type 'quit' to exit): ")

        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print(f"Error: File not found at {image_path}")
            continue

        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

            # Make a prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Get the predicted disease name
            predicted_disease = CLASS_INDEX_TO_DISEASE.get(str(predicted_class_index), "Unknown") # Use str() as keys in app.py are strings

            print("\n--- Prediction Results ---")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"  Predicted Class Index: {predicted_class_index}")
            print(f"  Predicted Disease: {predicted_disease}")
            print(f"  Confidence: {confidence:.4f}")
            print("------------------------\n")

        except FileNotFoundError:
             print(f"Error: File not found at {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path} or making prediction: {e}")

else:
    print("Model was not loaded, cannot perform predictions.")

print("Exiting model check script.")