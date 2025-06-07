from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import re
import requests
from twilio.rest import Client
import logging
from dotenv import load_dotenv
import phonenumbers
from utils import get_translation, is_valid_indian_phone_number
from constants import DISEASE_NUTRIENT_INTERACTIONS

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Twilio client
twilio_client = Client(
    os.getenv('TWILIO_ACCOUNT_SID'),
    os.getenv('TWILIO_AUTH_TOKEN')
)

# --- Model and Metadata Loading ---
model = None
model_metadata = None

# Hardcode the specific log directory from your successful training run
latest_log_dir = r'C:\Users\akbar\OneDrive\Documents\Kissancare\logs_debug\20250528_141305' # Use raw string for Windows path

if latest_log_dir and os.path.exists(latest_log_dir):
    best_model_path = os.path.join(latest_log_dir, 'best_model.h5') # Path to the best model file
    metadata_path = os.path.join(latest_log_dir, 'model_metadata.json') # Path to metadata

    model_loaded = False
    # Initialize these with fallbacks in case metadata loading fails
    loaded_CLASS_INDEX_TO_DISEASE = None
    loaded_MODEL_IMAGE_SIZE = 224
    loaded_CONFIDENCE_THRESHOLD = 0.5

    # Attempt to load metadata first
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Successfully loaded metadata from {metadata_path}")

            # Use metadata to populate variables
            # Ensure class names are loaded correctly, expecting a list of strings
            class_names_list = model_metadata.get('class_names', [])
            loaded_CLASS_INDEX_TO_DISEASE = {str(i): name for i, name in enumerate(class_names_list)} if class_names_list else None
            loaded_MODEL_IMAGE_SIZE = model_metadata.get('image_size', 224)
            loaded_CONFIDENCE_THRESHOLD = model_metadata.get('confidence_threshold', 0.5)
            logger.info(f"Using metadata: Image Size={loaded_MODEL_IMAGE_SIZE}, Confidence Threshold={loaded_CONFIDENCE_THRESHOLD}")

        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {str(e)}. Using hardcoded fallback values.")
            # Fallback will be handled below if loaded_CLASS_INDEX_TO_DISEASE is None
    else:
        logger.warning(f"Metadata not found at {metadata_path}. Using hardcoded fallback values.")
        # Fallback will be handled below if loaded_CLASS_INDEX_TO_DISEASE is None

    # Assign loaded or hardcoded fallback values globally
    CLASS_INDEX_TO_DISEASE = loaded_CLASS_INDEX_TO_DISEASE if loaded_CLASS_INDEX_TO_DISEASE else {
        str(i): f'Class_{i}' for i in range(38) # Fallback with generic class names if metadata fails completely
    } # Using string keys as per the metadata structure

    MODEL_IMAGE_SIZE = loaded_MODEL_IMAGE_SIZE
    CONFIDENCE_THRESHOLD = loaded_CONFIDENCE_THRESHOLD
    NUM_CLASSES = len(CLASS_INDEX_TO_DISEASE)

    # Attempt to load the best full model
    if os.path.exists(best_model_path):
        try:
            logger.info(f"Attempting to load best full model from {best_model_path}")
            # Configure TensorFlow to be less verbose during loading
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            # Load the entire model
            model = tf.keras.models.load_model(best_model_path)
            os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None) # Reset log level
            logger.info(f"Successfully loaded best full model from {best_model_path}")
            model_loaded = True
        except Exception as e_best:
            logger.error(f"Error loading best full model from {best_model_path}: {str(e_best)}. Using dummy model.")
    elif not model_loaded:
         logger.warning(f"Best full model not found at {best_model_path}. Using dummy model.")

    # If the best model was not loaded, create a dummy model
    if not model_loaded:
         # Create a dummy model for testing if loading fails
        logger.info("Creating dummy model for testing")
        # Ensure dummy model has the correct input shape and number of classes
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Use correct number of classes
        ])

else:
    logger.warning(f"Specified log directory not found: {latest_log_dir}. Using dummy model and hardcoded values.")
    # Fallback to hardcoded values and create dummy model if the specified log directory is not found
    # Define hardcoded class names (ensure they match your dataset if possible)
    CLASS_NAMES_HARDCODED = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
    CLASS_INDEX_TO_DISEASE = {str(i): name for i, name in enumerate(CLASS_NAMES_HARDCODED)}
    MODEL_IMAGE_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.5
    NUM_CLASSES = len(CLASS_INDEX_TO_DISEASE)

    # Create a dummy model if the specified log directory is not found
    logger.info("Creating dummy model as specified log directory not found")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # Use correct number of classes
    ])

# Define basic treatment recommendations for each disease
TREATMENT_RECOMMENDATIONS = {
    'Apple___Apple_scab': 'Remove infected leaves and fruit. Apply fungicides containing sulfur or copper.',
    'Apple___Black_rot': 'Prune out diseased branches. Apply fungicides after pruning and during wet periods.',
    'Apple___Cedar_apple_rust': 'Remove cedar trees within a few hundred feet. Apply fungicides containing myclobutanil or mancozeb.',
    'Apple___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Blueberry___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicides containing sulfur or potassium bicarbonate. Prune affected areas.',
    'Cherry_(including_sour)___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use resistant varieties. Apply fungicides if necessary.',
    'Corn_(maize)___Common_rust_': 'Use resistant varieties. Apply fungicides if necessary during early stages of infection.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties. Apply fungicides if necessary.',
    'Corn_(maize)___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Grape___Black_rot': 'Prune out diseased parts. Apply fungicides containing myclobutanil or mancozeb starting at bud break.',
    'Grape___Esca_(Black_Measles)': 'Prune out affected wood. There is no cure, focus on prevention.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides containing mancozeb or copper.',
    'Grape___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Orange___Haunglongbing_(Citrus_greening)': 'Remove infected trees. Control psyllid vectors with insecticides. No cure exists.',
    'Peach___Bacterial_spot': 'Use resistant varieties. Apply copper sprays during dormancy and antibiotic sprays during the growing season.',
    'Peach___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Pepper,_bell___Bacterial_spot': 'Use resistant varieties and disease-free seeds. Apply copper sprays.',
    'Pepper,_bell___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Potato___Early_blight': 'Rotate crops. Apply fungicides containing chlorothalonil or mancozeb.',
    'Potato___Late_blight': 'Use resistant varieties. Apply fungicides containing chlorothalonil, mancozeb, or phosphorus acid. Destroy infected plants.',
    'Potato___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Raspberry___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Soybean___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Squash___Powdery_mildew': 'Apply fungicides containing sulfur, neem oil, or potassium bicarbonate.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Apply fungicides containing copper or myclobutanil.',
    'Strawberry___healthy': 'Maintain good cultural practices, including proper watering and fertilization.',
    'Tomato___Bacterial_spot': 'Use resistant varieties and disease-free seeds. Apply copper sprays.',
    'Tomato___Early_blight': 'Rotate crops. Apply fungicides containing chlorothalonil or mancozeb.',
    'Tomato___Late_blight': 'Use resistant varieties. Apply fungicides containing chlorothalonil, mancozeb, or phosphorus acid. Destroy infected plants.',
    'Tomato___Leaf_Mold': 'Improve air circulation. Apply fungicides containing chlorothalonil or copper.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Apply fungicides containing chlorothalonil or mancozeb.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Hose down plants with water. Use insecticidal soaps or miticides.',
    'Tomato___Target_Spot': 'Rotate crops. Apply fungicides containing chlorothalonil or mancozeb.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whiteflies, which transmit the virus. Remove infected plants. No cure exists.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected plants and destroy them. Disinfect tools. No cure exists.',
    'Tomato___healthy': 'Maintain good cultural practices, including proper watering and fertilization.'
}

# Define nutrient interaction rules and recommendations
NUTRIENT_INTERACTIONS = {
    "Phosphorus-Zinc": {
        "condition": lambda nutrients: nutrients.get("phosphorus", {}).get("status") == "high" and nutrients.get("zinc", {}).get("status") != "excessive",
        "recommendation": {
            "priority": "medium",
            "action": "High Phosphorus levels might lead to Zinc deficiency. Consider a foliar application of Zinc.",
            "related_nutrients": ["phosphorus", "zinc"]
        }
    },
    "Nitrogen-Potassium Ratio": {
        "condition": lambda nutrients_values: nutrients_values.get("nitrogen") is not None and nutrients_values.get("potassium") is not None and nutrients_values["nitrogen"] / nutrients_values["potassium"] > 2, # Example: High N:K ratio
        "recommendation": {
            "priority": "medium",
            "action": "High Nitrogen to Potassium ratio detected. Ensure adequate Potassium supply to balance Nitrogen uptake.",
            "related_nutrients": ["nitrogen", "potassium"]
        }
    }
}

# Define nutrient thresholds
NUTRIENT_THRESHOLDS = {
    'nitrogen': {'low': 40, 'high': 80},
    'phosphorus': {'low': 20, 'high': 40},
    'potassium': {'low': 30, 'high': 60},
    'magnesium': {'low': 20, 'high': 40},
    'calcium': {'low': 30, 'high': 60},
    'sulfur': {'low': 10, 'high': 30}
}

# Define crop-specific nutrient ranges (kg/ha)
CROP_NUTRIENT_RANGES = {
    "Apple": {
        "Nitrogen": (30, 50),
        "Phosphorus": (15, 25),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (150, 200),
        "Sulfur": (10, 20)
    },
    "Blueberry": {
        "Nitrogen": (20, 40),
        "Phosphorus": (10, 20),
        "Potassium": (40, 80),
        "Magnesium": (10, 20),
        "Calcium": (50, 100),
        "Sulfur": (5, 15)
    },
    "Cherry": {
        "Nitrogen": (30, 60),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (15, 30),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Corn": {
        "Nitrogen": (100, 150),
        "Phosphorus": (30, 50),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (15, 30)
    },
    "Grape": {
        "Nitrogen": (30, 60),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Orange": {
        "Nitrogen": (60, 100),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Peach": {
        "Nitrogen": (40, 80),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Pepper, bell": {
        "Nitrogen": (80, 120),
        "Phosphorus": (30, 50),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Potato": {
        "Nitrogen": (100, 150),
        "Phosphorus": (30, 50),
        "Potassium": (150, 200),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (15, 30)
    },
    "Raspberry": {
        "Nitrogen": (30, 60),
        "Phosphorus": (20, 40),
        "Potassium": (80, 120),
        "Magnesium": (15, 30),
        "Calcium": (80, 120),
        "Sulfur": (10, 20)
    },
    "Soybean": {
        "Nitrogen": (0, 0),  # Soybeans fix atmospheric nitrogen
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Squash": {
        "Nitrogen": (60, 100),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Strawberry": {
        "Nitrogen": (60, 100),
        "Phosphorus": (20, 40),
        "Potassium": (100, 150),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    },
    "Tomato": {
        "Nitrogen": (100, 150),
        "Phosphorus": (30, 50),
        "Potassium": (150, 200),
        "Magnesium": (20, 40),
        "Calcium": (100, 150),
        "Sulfur": (10, 20)
    }
}

# Define language translations
translations = {
    'en': {
        'app_name': 'KisanCare',
        'home_title': 'Welcome to KisanCare',
        'basic_mode_title': 'Basic Mode',
        'advanced_mode_title': 'Advanced Mode',
        'select_language': 'Select Language',
        'basic_mode_description': 'Upload an image of a plant leaf to get disease prediction.',
        'advanced_mode_description': 'Upload an image and enter soil nutrient data for combined analysis.',
        'get_started': 'Get Started',
        'home_page_title': 'KisanCare - Home',
        'basic_mode_title_page': 'KisanCare - Basic Mode',
        'advanced_mode_title_page': 'KisanCare - Advanced Mode',
        'quick_guide_title': 'Quick Guide',
        'quick_guide_step1': 'Select your preferred language.',
        'quick_guide_step2': 'Choose Basic or Advanced mode.',
        'quick_guide_step3': 'Follow instructions on the next page.',
        'quick_guide_title_advanced': 'Advanced Mode Quick Guide',
        'quick_guide_step1_advanced': 'Enter phone number (optional) for SMS results.',
        'quick_guide_step2_advanced': 'Enter soil nutrient levels.',
        'quick_guide_step3_advanced': 'Upload plant leaf image and analyze.',
        'phone_number_label': 'Your Phone Number (Optional for SMS)',
        'phone_number_help': 'Enter your phone number to receive diagnosis via SMS.',
        'upload_image_label': 'Upload Plant Leaf Image',
        'image_upload_help': 'Upload a clear image of the affected plant leaf.',
        'capture_photo': 'Capture Photo',
        'analyze_button': 'Analyze',
        'tips_title': 'Tips for Best Results',
        'tip_1': 'Use clear, focused images.',
        'tip_2': 'Ensure good lighting.',
        'tip_3': 'Focus on the affected area.',
        'tip_4': 'One leaf per image for basic mode.',
        'image_preview_alt': 'Image Preview',
        'analyzing_message': 'Analyzing...',
        'please_wait_message': 'Please wait while we analyze the image.',
        'analysis_complete_message': 'Analysis Complete',
        'prediction_class_prefix': 'Predicted Class:',
        'confidence_prefix': 'Confidence:',
        'basic_treatment_guidance_heading': 'Basic Treatment Guidance:',
        'sms_sent_message': 'Results sent via SMS (simulated). Check console for message.',
        'sms_send_error': 'Failed to send SMS.',
        'unknown_error': 'An unknown error occurred.',
        'form_submit_error': 'Error submitting form. Please try again.',
        'camera_error': 'Could not access camera.',
        'error': 'Error',
        'camera_not_supported': 'Camera capture not supported by your browser.',
        'no_image_uploaded': 'No image file uploaded.',
        'invalid_file_type': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif).',
        'empty_image_file': 'Uploaded image file is empty.',
        'image_preprocess_error': 'Error processing image.',
        'prediction_error': 'Error during prediction.',
        'low_confidence_prediction': 'Could not make a confident prediction. Please try another image.',
        'nutrient_analysis_heading': 'Nutrient Analysis:',
        'recommendations_heading': 'Recommendations:',
        'nutrient_analysis_error': 'Error during nutrient analysis.',
        'high_priority_marker': '[HIGH PRIORITY]',
        'invalid_phone_number': 'Invalid phone number format.',
        'nutrient_status_optimal': 'Status: Optimal',
        'nutrient_status_deficient': 'Status: Deficient',
        'nutrient_status_excessive': 'Status: Excessive',
        'nutrient_status_mixed': 'Status: Mixed',
        'nutrient_status_invalid': 'Status: Invalid Data',
        'nutrient_status_error': 'Status: Processing Error',
        'nutrient_input_heading': 'Enter Soil Nutrient Levels (kg/ha)',
        'nitrogen_label': 'Nitrogen',
        'phosphorus_label': 'Phosphorus',
        'potassium_label': 'Potassium',
        'magnesium_label': 'Magnesium',
        'calcium_label': 'Calcium',
        'sulfur_label': 'Sulfur',
        'nutrient_input_help': 'Enter the levels of key nutrients in your soil.',
        'advanced_analysis_heading': 'Advanced Analysis Results',
        'nutrient_levels_heading': 'Nutrient Levels',
        'value_label': 'Value',
        'status_label': 'Status',
        'target_range_label': 'Target Range',
        'phone_number_tooltip': 'Enter your 10-digit Indian mobile number with +91 prefix.',
        'invalid_phone_format': 'Please enter a valid Indian phone number starting with +91 followed by 10 digits.',
        'nutrient_details_heading': 'Enter Soil Nutrient Levels (kg/ha)',
        'nutrient_details_help': 'Enter the levels of key nutrients in your soil.',
        'invalid_nutrient_format': 'Please enter valid numeric values for nutrient levels.',
        'capture_photo_aria': 'Capture photo using camera',
        'remove_image_aria': 'Remove uploaded image'
    },
    'te': {
        'app_name': 'కిసాన్‌కేర్',
        'home_title': 'కిసాన్‌కేర్‌కు స్వాగతం',
        'basic_mode_title': 'బేసిక్ మోడ్',
        'advanced_mode_title': 'అడ్వాన్స్‌డ్ మోడ్',
        'select_language': 'భాషను ఎంచుకోండి',
        'basic_mode_description': 'మొక్కల ఆకుల చిత్రాన్ని అప్‌లోడ్ చేసి వ్యాధిని గుర్తించండి.',
        'advanced_mode_description': 'చిత్రాన్ని అప్‌లోడ్ చేసి, మట్టిలోని పోషకాల వివరాలు నమోదు చేయండి.',
        'get_started': 'ప్రారంభించండి',
        'home_page_title': 'కిసాన్‌కేర్ - హోమ్',
        'basic_mode_title_page': 'కిసాన్‌కేర్ - బేసిక్ మోడ్',
        'advanced_mode_title_page': 'కిసాన్‌కేర్ - అడ్వాన్స్‌డ్ మోడ్',
        'quick_guide_title': 'త్వరిత గాఇడ్',
        'quick_guide_step1': 'మీకు నచ్చిన భాషను ఎంచుకోండి.',
        'quick_guide_step2': 'బేసిక్ లేదా అడ్వాన్స్‌డ్ మోడ్ ఎంచుకోండి.',
        'quick_guide_step3': 'తదుపరి పేజీలో సూచనలను అనుసరించండి.',
        'quick_guide_title_advanced': 'అడ్వాన్స్‌డ్ మోడ్ త్వరిత గాఇడ్',
        'quick_guide_step1_advanced': 'SMS ఫలితాల కోసం ఫోన్ నంబర్ (ఐచ్ఛికం) నమోదు చేయండి.',
        'quick_guide_step2_advanced': 'మట్టి పోషక స్థాయిలను నమోదు చేయండి.',
        'quick_guide_step3_advanced': 'మొక్కల ఆకు చిత్రాన్ని అప్‌లోడ్ చేసి విశ్లేషించండి.',
        'phone_number_label': 'మీ ఫోన్ నంబర్ (SMS కొరకు ఐచ్ఛికం)',
        'phone_number_help': 'SMS ద్వారా రోగ నిర్ధారణ పొందడానికి మీ ఫోన్ నంబర్ నమోదు చేయండి.',
        'upload_image_label': 'మొక్కల ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి',
        'image_upload_help': 'వ్యాధి సోకిన మొక్క ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్‌లోడ్ చేయండి.',
        'capture_photo': 'ఫోటో తీయండి',
        'analyze_button': 'విశ్లేషించండి',
        'tips_title': 'ఉత్తమ ఫలితాల కోసం చిట్కాలు',
        'tip_1': 'స్పష్టమైన, ఫోకస్ చేసిన చిత్రాలను ఉపయోగించండి.',
        'tip_2': 'మంచి వెలుతురు ఉండేలా చూసుకోండి.',
        'tip_3': 'వ్యాధి సోకిన ప్రదేశంపై దృష్టి పెట్టండి.',
        'tip_4': 'బేసిక్ మోడ్ కొరకు ప్రతి చిత్రంలో ఒక ఆకు మాత్రమే ఉండేలా చూడండి.',
        'image_preview_alt': 'చిత్ర ప్రివ్యూ',
        'analyzing_message': 'విశ్లేషిస్తోంది...',
        'please_wait_message': 'మేము చిత్రాన్ని విశ్లేషించే వరకు వేచి ఉండండి.',
        'analysis_complete_message': 'విశ్లేషణ పూర్తయింది',
        'prediction_class_prefix': 'ఊహించిన తరగతి:',
        'confidence_prefix': 'విశ్వాసం:',
        'basic_treatment_guidance_heading': 'ప్రాథమిక చికిత్స మార్గదర్శకం:',
        'sms_sent_message': 'ఫలితాలు SMS ద్వారా భంపబడ్డాయి (అనుకరణ). మెసేజ్ కొరకు కన్సోల్ చూడండి.',
        'sms_send_error': 'SMS భంపడంలో విఫలమైంది.',
        'unknown_error': 'తెలియని లోపం సంభవింది.',
        'form_submit_error': 'ఫారం సమర్పించడంలో లోపం. దయచేసి మళ్ళీ ప్రయత్నించండి.',
        'camera_error': 'కెమెరాను యాక్సెస్ చేయలేకపోయింది.',
        'error': 'లోపం',
        'camera_not_supported': 'మీ బ్రాఉజర్ కెమెరా క్యాప్చర్‌ను సపోర్ట్ చేయదు.',
        'no_image_uploaded': 'చిత్ర ఫాఇల్ అప్‌లోడ్ చేయబడలేదు.',
        'invalid_file_type': 'చెల్లని ఫాఇల్ రకం. దయచేసి చిత్రాన్ని అప్‌లోడ్ చేయండి (png, jpg, jpeg, gif).',
        'empty_image_file': 'అప్‌లోడ్ చేసిన చిత్ర ఫాఇల్ ఖాళీగా ఉంది.',
        'image_preprocess_error': 'చిత్రాన్ని ప్రాసెస్ చేయడంలో లోపం.',
        'prediction_error': 'ఊహించడంలో లోపం.',
        'low_confidence_prediction': 'ఖచ్చితమైన ఊహించలేకపోయింది. దయచేసి మరొక చిత్రాన్ని ప్రయత్నించండి.',
        'nutrient_analysis_heading': 'పోషక విశ్లేషణ:',
        'recommendations_heading': 'సూచనలు:',
        'nutrient_analysis_error': 'పోషక విశ్లేషణలో లోపం.',
        'high_priority_marker': '[అత్యంత ప్రాధాన్యత]',
        'invalid_phone_number': 'చెల్లని ఫార్మాట్.',
        'nutrient_status_optimal': 'స్థితి: సరైనది',
        'nutrient_status_deficient': 'స్థితి: లోపం',
        'nutrient_status_excessive': 'స్థితి: అధికం',
        'nutrient_status_mixed': 'స్థితి: మిశ్రమ',
        'nutrient_status_invalid': 'స్థితి: అమాన్య డేటా',
        'nutrient_status_error': 'స్థితి: ప్రసంస్కరణ లోపం',
        'nutrient_input_heading': 'మట్టి పోషక స్థాయిలు నమోదు చేయండి (కిగ్రా/హేక్టేయర)',
        'nitrogen_label': 'నాయిట్రోని',
        'phosphorus_label': 'ఫాస్ఫోరస్',
        'potassium_label': 'పోటాషియం',
        'magnesium_label': 'మగ్నీషియం',
        'calcium_label': 'కాల్షియం',
        'sulfur_label': 'సల్ఫర్',
        'nutrient_input_help': 'మీ మట్టిలోని ముఖ్య పోషక స్థాయిలను నమోదు చేయండి.',
        'advanced_analysis_heading': 'అడ్వాన్స్‌డ్ విశ్లేషణ ఫలితాలు',
        'nutrient_levels_heading': 'పోషక స్థాయిలు',
        'value_label': 'విలువ',
        'status_label': 'స్థితి',
        'target_range_label': 'లక్ష్య పరిధి',
        'phone_number_tooltip': '+91తో భంపబడే మీ 10-అంకెల ఇందియన్ మొబైల్ నంబర్ నమోదు చేయండి.',
        'invalid_phone_format': '+91తో భంపబడే 10-అంకెల చెల్లుబాటు అయ్యే ఇందియన్ ఫోన్ నంబర్ నమోదు చేయండి.',
        'nutrient_details_heading': 'మట్టి పోషక స్థాయిలు నమోదు చేయండి (కిగ్రా/హేక్టేయర)',
        'nutrient_details_help': 'మీ మట్టిలోని ముఖ్య పోషక స్థాయిలను నమోదు చేయండి.',
        'invalid_nutrient_format': 'పోషక స్థాయిల కోసం చెల్లుబాటు అయ్యే సంఖ్యాత్మక విలువల నమోదు చేయండి.',
        'capture_photo_aria': 'కెమెరా ఉపయోగించి ఫోటో తీయండి',
        'remove_image_aria': 'అప్‌లోడ్ చేసిన చిత్రాన్ని తీసివేయండి'
    },
    'hi': {
        'app_name': 'किसानकेयर',
        'home_title': 'किसानकेयर में आपका स्वागत है',
        'basic_mode_title': 'बेसिक मोड',
        'advanced_mode_title': 'एडवांस्ड मोड',
        'select_language': 'भाषा चुनें',
        'basic_mode_description': 'रोग की पहचान के लिए पौधे की पत्ती की तस्वीर अपलोड करें।',
        'advanced_mode_description': 'तस्वीर अपलोड करें और संयुक्त विश्लेषण के लिए मिट्टी के पोषक तत्वों का डेटा दर्ज करें।',
        'get_started': 'शुरू करें',
        'home_page_title': 'किसानकेयर - होम',
        'basic_mode_title_page': 'किसानकेयर - बेसिक मोड',
        'advanced_mode_title_page': 'किसानकेयर - एडवांस्ड मोड',
        'quick_guide_title': 'त्वरित गाइड',
        'quick_guide_step1': 'अपनी पसंदीदा भाषा चुनें।',
        'quick_guide_step2': 'बేजइ या एडवांस्ड मोड चुनें।',
        'quick_guide_step3': 'अगले पृष्ठ पर दिए गए निर्देशों का पालन करें।',
        'quick_guide_title_advanced': 'एडवांस्ड मोड त्वरित गाइड',
        'quick_guide_step1_advanced': 'SMS परिणामों के लिए फ़ोन नंबर (वैकल्पिक) दर्ज करें।',
        'quick_guide_step2_advanced': 'मिट्टी के पोषक तत्व स्तर दर्ज करें।',
        'quick_guide_step3_advanced': 'पौधे की पत्ती की तस्वीर अपलोड करें और विश्लेषण करें।',
        'phone_number_label': 'आपका फ़ोन नंबर (SMS के लिए वैकल्पिक)',
        'phone_number_help': 'SMS के माध्यम से निदान प्राप्त करने के लिए अपना फ़ोन नंबर दर्ज करें।',
        'upload_image_label': 'पौधे की पत्ती की तस्वीर अपलोड करें',
        'image_upload_help': 'रोगग्रस्त पौधे की पत्ती की एक स्पष्ट तस्वीर अपलोड करें।',
        'capture_photo': 'फोटो लें',
        'analyze_button': 'विश्लेषण करें',
        'tips_title': 'सर्वोत्तम परिणामों के लिए सुझाव',
        'tip_1': 'स्पष्ट, केंद्रित छवियों का उपयोग करें।',
        'tip_2': 'सुनिश्चित करें कि प्रकाश व्यवस्था अच्छी हो।',
        'tip_3': 'प्रभावित क्षेत्र पर ध्यान केंद्रित करें।',
        'tip_4': 'बेसिक मोड के लिए प्रति छवि एक पत्ती।',
        'image_preview_alt': 'छवि पूर्वावलोकन',
        'analyzing_message': 'विश्लेषण हो रहा है...',
        'please_wait_message': 'जब तक हम छवि का विश्लेषण करें तब तक प्रतीक्षा करें।',
        'analysis_complete_message': 'विश्लेषण पूर्ण',
        'prediction_class_prefix': 'अनुमानित वर्ग:',
        'confidence_prefix': 'आत्मविश्वास:',
        'basic_treatment_guidance_heading': 'बुनियादी उपचार मार्गदर्शन:',
        'sms_sent_message': 'परिणाम SMS के माध्यम से भेजे गए (सिमुलेटेड)। संदेश के लिए कंसोल जांचें।',
        'sms_send_error': 'SMS भेजने में विफल।',
        'unknown_error': 'एक अज्ञात त्रुटि हुई।',
        'form_submit_error': 'फॉर्म सबमिट करने में त्रुटि। कृपया पुनः प्रयास करें।',
        'camera_error': 'कैमरे तक पहुँचने में विफल।',
        'error': 'त्रुटि',
        'camera_not_supported': 'आपका ब्राउज़र कैमरा कैप्चर का समर्थन नहीं करता है।',
        'no_image_uploaded': 'कोई छवि फ़ाइल अपलोड नहीं की गई।',
        'invalid_file_type': 'अमान्य फ़ाइल प्रकार। कृपया एक छवि अपलोड करें (png, jpg, jpeg, gif)।',
        'empty_image_file': 'अपलोड की गई छवि फ़ाइल खाली है।',
        'image_preprocess_error': 'छवि को संसाधित करने में त्रुटि।',
        'prediction_error': 'पूर्वानुमान के दौरान त्रुटि।',
        'low_confidence_prediction': 'आत्मविश्वासी पूर्वानुमान नहीं लगाया जा सका। कृपया कोई अन्य छवि प्रयास करें।',
        'nutrient_analysis_heading': 'पोषक तत्व विश्लेषण:',
        'recommendations_heading': 'सुझाव:',
        'nutrient_analysis_error': 'पोषक तत्व विश्लेषण के दौरान त्रुटि।',
        'high_priority_marker': '[उच्च प्राथमिकता]',
        'invalid_phone_number': 'अमान्य फ़ोन नंबर प्रारूप।',
        'nutrient_status_optimal': 'स्थिति: इष्टतम',
        'nutrient_status_deficient': 'स्थिति: कमी',
        'nutrient_status_excessive': 'स्थिति: अत्यधिक',
        'nutrient_status_mixed': 'स्थिति: मिश्रित',
        'nutrient_status_invalid': 'स्थिति: अमान्य डेटा',
        'nutrient_status_error': 'स्थिति: प्रसंस्करण त्रुटि',
        'nutrient_input_heading': 'मिट्टी के पोषक तत्व स्तर दर्ज करें (किग्रा/हेक्टेयर)',
        'nitrogen_label': 'नाइट्रोजन',
        'phosphorus_label': 'फास्फोरस',
        'potassium_label': 'पोटेशियम',
        'magnesium_label': 'मैग्नीशियम',
        'calcium_label': 'कैल्शियम',
        'sulfur_label': 'सल्फर',
        'nutrient_input_help': 'अपनी मिट्टी में मुख्य पोषक तत्वों का स्तर दर्ज करें।',
        'advanced_analysis_heading': 'एडवांस्ड विश्लेषण परिणाम',
        'nutrient_levels_heading': 'पोषक तत्व स्तर',
        'value_label': 'मान',
        'status_label': 'स्थिति',
        'target_range_label': 'लक्ष्य सीमा',
        'phone_number_tooltip': '+91 के साथ अपना 10 अंकों का भारतीय मोबाइल नंबर दर्ज करें।',
        'invalid_phone_format': 'कृपया +91 के साथ 10 अंकों का मान्य भारतीय फ़ोन नंबर दर्ज करें।',
        'nutrient_details_heading': 'मिट्टी के पोषक तत्व स्तर दर्ज करें (किग्रा/हेक्टेयर)',
        'nutrient_details_help': 'अपनी मिट्टी में मुख्य पोषक तत्वों का स्तर दर्ज करें।',
        'invalid_nutrient_format': 'कृपया पोषक तत्व स्तरों के लिए मान्य संख्यात्मक मान दर्ज करें।',
        'capture_photo_aria': 'कैमरे का उपयोग करके फोटो कैप्चर करें',
        'remove_image_aria': 'अपलोड की गई छवि हटाएं'
    }
}

LANGUAGES = translations.keys()

def sanitize_input(input_str):
    # Implement sanitization logic here
    return input_str

def preprocess_image(image_file):
    """Preprocesses the uploaded image for model prediction."""
    try:
        img = Image.open(image_file).convert('RGB')
        # Use MODEL_IMAGE_SIZE loaded from metadata or fallback
        img = img.resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
        img_array = tf.keras.utils.img_to_array(img)
        return img_array
    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        return None

def analyze_nutrients(nutrient_data, crop_name=None, disease_name=None):
    """Analyzes nutrient levels and provides recommendations."""
    analysis = {
        'nutrients': {},
        'overall_status': 'optimal', # Added overall status
        'recommendations': []
    }
    
    # Use crop-specific ranges if available, otherwise use general thresholds
    thresholds_to_use = CROP_NUTRIENT_RANGES.get(crop_name, NUTRIENT_THRESHOLDS)
    is_crop_specific = crop_name in CROP_NUTRIENT_RANGES

    has_low_nutrients = False
    has_high_nutrients = False
    high_priority_count = 0
    low_priority_count = 0

    for nutrient, value_str in nutrient_data.items():
        # Convert nutrient name to lowercase for consistent dictionary lookup
        nutrient_lower = nutrient.lower()
        
        if nutrient_lower in thresholds_to_use:
            try:
                value = float(value_str)
                analysis['nutrients'][nutrient_lower] = {'value': value, 'status': 'optimal'}

                thresholds = thresholds_to_use[nutrient_lower]

                # Determine status based on thresholds
                if is_crop_specific:
                    low_threshold, high_threshold = thresholds
                    if value < low_threshold:
                        analysis['nutrients'][nutrient_lower]['status'] = 'low'
                        has_low_nutrients = True
                        analysis['recommendations'].append({
                            'priority': 'high',
                            'action': f'Apply {nutrient.capitalize()} fertilizer',
                            'target_range': f'Target: {low_threshold}-{high_threshold}'
                        })
                        high_priority_count += 1
                    elif value > high_threshold:
                        analysis['nutrients'][nutrient_lower]['status'] = 'high'
                        has_high_nutrients = True
                        analysis['recommendations'].append({
                             'priority': 'medium',
                            'action': f'Reduce {nutrient.capitalize()} application or consider flushing',
                            'target_range': f'Target: {low_threshold}-{high_threshold}'
                        })
                        low_priority_count += 1
                else: # Use general thresholds
                     low_threshold = thresholds.get('low')
                     high_threshold = thresholds.get('high')

                     if low_threshold is not None and value < low_threshold:
                         analysis['nutrients'][nutrient_lower]['status'] = 'low'
                         has_low_nutrients = True
                         analysis['recommendations'].append({
                             'priority': 'high',
                             'action': f'Apply {nutrient.capitalize()} fertilizer',
                             'target_range': f'Below threshold: {low_threshold}'
                         })
                         high_priority_count += 1
                     elif high_threshold is not None and value > high_threshold:
                         analysis['nutrients'][nutrient_lower]['status'] = 'high'
                         has_high_nutrients = True
                         analysis['recommendations'].append({
                              'priority': 'medium',
                             'action': f'Reduce {nutrient.capitalize()} application or consider flushing',
                             'target_range': f'Above threshold: {high_threshold}'
                         })
                         low_priority_count += 1

            except ValueError:
                analysis['nutrients'][nutrient_lower] = {'value': value_str, 'status': 'invalid'}
                analysis['recommendations'].append({
                    'priority': 'high',
                    'action': f'Invalid value provided for {nutrient.capitalize()}.',
                    'target_range': 'N/A'
                    })
                high_priority_count += 1
            except Exception as e:
                logger.error(f"Error processing nutrient {nutrient}: {str(e)}")
                analysis['nutrients'][nutrient_lower] = {'value': value_str, 'status': 'error'}
                analysis['recommendations'].append({
                    'priority': 'high',
                    'action': f'Error processing data for {nutrient.capitalize()}.',
                    'target_range': 'N/A'
                })
                high_priority_count += 1

    # After initial nutrient status analysis, check for general nutrient interactions
    nutrient_values = {nut: data['value'] for nut, data in analysis['nutrients'].items() if data['status'] != 'invalid' and data['status'] != 'error'}
    for interaction_name, interaction_rule in NUTRIENT_INTERACTIONS.items():
        # Pass the full analysis['nutrients'] (with status) and values to the condition lambda
        try:
            # Check if the lambda condition is met. It receives analysis['nutrients'] and nutrient_values
            if interaction_rule["condition"](analysis['nutrients'], nutrient_values):
                # Add recommendation if the condition is met and it's not already added (to avoid duplicates)
                recommendation_to_add = interaction_rule["recommendation"]
                if recommendation_to_add not in analysis['recommendations']:
                    analysis['recommendations'].append(recommendation_to_add)
                    if recommendation_to_add.get('priority') == 'high':
                         high_priority_count += 1
                    else:
                         low_priority_count += 1
        except Exception as e:
             logger.error(f"Error checking general nutrient interaction {interaction_name}: {str(e)}")

    # If a disease is predicted, check for disease-specific nutrient interactions
    if disease_name and disease_name in DISEASE_NUTRIENT_INTERACTIONS:
        disease_interactions = DISEASE_NUTRIENT_INTERACTIONS[disease_name]
        for interaction in disease_interactions:
            try:
                 # Check if the lambda condition is met. It receives analysis['nutrients']
                 if interaction["condition"](analysis['nutrients']):
                     # Add recommendation if the condition is met and it's not already added
                     recommendation_to_add = interaction["recommendation"]
                     if recommendation_to_add not in analysis['recommendations']:
                          analysis['recommendations'].append(recommendation_to_add)
                          if recommendation_to_add.get('priority') == 'high':
                               high_priority_count += 1
                          else:
                               low_priority_count += 1
            except Exception as e:
                 logger.error(f"Error checking disease-nutrient interaction for {disease_name}: {str(e)}")

    # Determine overall status
    if has_low_nutrients and has_high_nutrients:
        analysis['overall_status'] = 'mixed'
    elif has_low_nutrients:
        analysis['overall_status'] = 'deficient'
    elif has_high_nutrients:
        analysis['overall_status'] = 'excessive'
    else:
        analysis['overall_status'] = 'optimal'

    # Add a summary based on the number and type of issues
    summary = None
    total_issues = high_priority_count + low_priority_count
    if total_issues > 1:
        if has_low_nutrients and has_high_nutrients:
            summary = "Multiple nutrient issues detected including both deficiencies and excesses." # Mixed issues
        elif has_low_nutrients:
             summary = f"Multiple nutrient deficiencies detected ({high_priority_count} high priority)."
        elif has_high_nutrients:
             summary = f"Multiple nutrient excesses detected ({low_priority_count} medium priority)."
        else: # Should cover interaction issues if no low/high nutrients
             summary = "Potential nutrient interactions detected."
    elif total_issues == 1:
         if has_low_nutrients:
              summary = f"One high priority nutrient deficiency detected."
         elif has_high_nutrients:
              summary = f"One medium priority nutrient excess detected."
         else:
              summary = "One potential nutrient interaction detected."

    if summary:
        # Add the summary as the first recommendation with a neutral priority
        analysis['recommendations'].insert(0, {
            'priority': 'info', # Use 'info' or a similar level for summaries
            'action': summary,
            'target_range': 'N/A'
        })

    # Sort recommendations by priority (high first), then info, then medium
    priority_order = {'high': 0, 'info': 1, 'medium': 2, 'low': 3}
    analysis['recommendations'].sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 4))
    
    return analysis

def send_sms(phone_number, message):
    """Sends an SMS using Twilio"""
    try:
        # In a real scenario, you'd uncomment this to send the SMS
        # twilio_client.messages.create(
        #     to=phone_number,
        #     from_=os.getenv('TWILIO_PHONE_NUMBER'), # Your Twilio phone number
        #     body=message
        # )
        logger.info(f"SMS simulated for {phone_number}: {message}")
        return True, "SMS simulated successfully."
    except Exception as e:
        logger.error(f"Failed to send SMS to {phone_number}: {str(e)}")
        return False, f"Failed to send SMS: {str(e)}"+"\n"+message # Include message in error for debugging

@app.route('/')
def index():
    lang = request.args.get('lang', 'en')
    return render_template('index.html', text=translations.get(lang, translations['en']), current_lang=lang, languages=list(LANGUAGES))

@app.route('/basic_mode')
def basic_mode():
    lang = request.args.get('lang', 'en')
    return render_template('basic_mode.html', text=translations.get(lang, translations['en']), current_lang=lang, languages=list(LANGUAGES))

@app.route('/advanced_mode')
def advanced_mode():
    lang = request.args.get('lang', 'en')
    return render_template('advanced_mode.html', text=translations.get(lang, translations['en']), current_lang=lang, languages=list(LANGUAGES))

@app.route('/predict', methods=['POST'])
def predict():
    lang = request.args.get('lang', 'en')
    # Ensure a file was uploaded
    if 'image' not in request.files:
        logger.warning("No image file part in the request")
        return jsonify({'error': get_translation('no_image_uploaded', lang)}), 400

    file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        logger.warning("No selected image file")
        return jsonify({'error': get_translation('no_image_uploaded', lang)}), 400

    if file:
        try:
            # Preprocess the image
            processed_image = preprocess_image(file)

            if processed_image is None:
                logger.error("Image preprocessing failed.")
                return jsonify({'error': get_translation('image_preprocess_error', lang)}), 500

            # Make prediction
            # Expand dimensions to match model input shape (batch size 1)
            input_tensor = np.expand_dims(processed_image, axis=0)

            if model is None:
                logger.error("Model is not loaded.")
                # Fallback to dummy model or return error - currently using dummy
                # return jsonify({'error': 'Model not loaded'}), 500 # Alternative

            predictions = model.predict(input_tensor)

            # Get the predicted class index and confidence
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Check confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                logger.info(f"Prediction confidence ({confidence:.2f}) below threshold ({CONFIDENCE_THRESHOLD}).")
                return jsonify({'status': 'warning', 'message': get_translation('low_confidence_prediction', lang)}), 400

            # Get the disease name using the mapping
            # Ensure the predicted_class_index is treated as an int as the dictionary keys are ints
            predicted_disease = CLASS_INDEX_TO_DISEASE.get(str(predicted_class_index), get_translation('unknown_error', lang))

            # Get the treatment recommendation (Basic Mode)
            treatment = TREATMENT_RECOMMENDATIONS.get(predicted_disease, get_translation('unknown_error', lang))

            logger.info(f"Prediction: {predicted_disease}, Confidence: {confidence:.2f}")

            # Get mode and phone number from form data
            mode = request.form.get('mode', 'basic')
            phone_number = request.form.get('phone')
            sms_status_message = ""
            nutrient_analysis_result = None

            if mode == 'advanced':
                 # 4. Nutrient Analysis (Advanced Mode Only)
                 nutrient_data = {
                     'nitrogen': request.form.get('nitrogen'),
                     'phosphorus': request.form.get('phosphorus'),
                     'potassium': request.form.get('potassium'),
                     'magnesium': request.form.get('magnesium'),
                     'calcium': request.form.get('calcium'),
                     'sulfur': request.form.get('sulfur'),
                     # Add other nutrients as needed
                 }

                 # Extract crop name from disease prediction for specific recommendations
                 # Assumes disease name format like 'CropName___DiseaseName'
                 crop_name = predicted_disease.split('___')[0] if '___' in predicted_disease else None

                 try:
                     # Pass the predicted disease name to analyze_nutrients
                     nutrient_analysis_result = analyze_nutrients(nutrient_data, crop_name, predicted_disease)
                 except Exception as e:
                     logger.error(f"Error during nutrient analysis: {str(e)}")
                     # Continue with disease prediction results but indicate nutrient analysis failure
                     nutrient_analysis_result = {'error': get_translation('nutrient_analysis_error', lang), 'message': str(e)}

            if phone_number:
                # Validate phone number
                if not is_valid_indian_phone_number(phone_number):
                    logger.warning(f"Invalid phone number format: {phone_number}")
                    # Continue without sending SMS, but inform the user
                    sms_status_message = get_translation('invalid_phone_number', lang) + ". " + get_translation('sms_send_error', lang)
                else:
                    # Prepare SMS message (consider including nutrient analysis results if available)
                    sms_message_body = f"KisanCare Diagnosis: Disease - {predicted_disease}, Confidence - {confidence:.2f}. Treatment - {treatment}."

                    if nutrient_analysis_result and 'recommendations' in nutrient_analysis_result:
                         sms_message_body += " Nutrient Analysis: " + nutrient_analysis_result.get('overall_status', 'N/A') + ". Recommendations: "
                         for rec in nutrient_analysis_result.get('recommendations', [])[:3]: # Limit recommendations in SMS
                              sms_message_body += f" - {rec['action']}."

                    try:
                        # Send SMS (this is simulated/commented out in actual code)
                        # send_sms(phone_number, sms_message_body)
                        logger.info(f"SMS simulation: Sending to {phone_number} with message: {sms_message_body}")
                        sms_status_message = get_translation('sms_sent_message', lang)
                    except Exception as sms_e:
                        logger.error(f"Failed to send SMS to {phone_number}: {str(sms_e)}")
                        sms_status_message = get_translation('sms_send_error', lang)

            # Return the prediction result, confidence, treatment, and SMS status as JSON
            response_data = {
                'disease': predicted_disease,
                'confidence': confidence,
                'treatment': treatment, # Basic mode treatment always included
                'message': get_translation('analysis_complete_message', lang),
            }
            if sms_status_message:
                 response_data['sms_status'] = sms_status_message

            if mode == 'advanced' and nutrient_analysis_result:
                 response_data['nutrient_analysis'] = nutrient_analysis_result # Include full analysis result in advanced mode

            return jsonify(response_data), 200

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True) # Log exception details
            return jsonify({'error': get_translation('prediction_error', lang), 'message': f"An error occurred: {type(e).__name__} - {str(e)}"}), 500 # Include exception type in message

    # Should not reach here if file is present, but as a fallback
    logger.warning("Reached end of predict function without processing file.")
    return jsonify({'error': get_translation('unknown_error', lang)}), 500

if __name__ == '__main__':
    # In a college project demo, running with debug=True is acceptable.
    # For production, use a production WSGI server like Gunicorn or uWSGI.
    app.run(debug=True)