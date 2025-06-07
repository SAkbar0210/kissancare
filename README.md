# KisanCare - Smart Crop Disease Detection

KisanCare is an intelligent agricultural assistance application that helps farmers identify crop diseases and deficiencies using AI technology. The application provides both basic and advanced modes of operation, with features like SMS notifications and nutrient analysis.

## Features

### Basic Mode
- Simple photo upload for disease detection
- Instant AI-powered diagnosis
- Basic treatment guidance
- SMS report delivery

### Advanced Mode
- Detailed disease analysis
- Nutrient level analysis from PDF reports
- Scientific insights and recommendations
- Comprehensive SMS reports

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for PDF text extraction)
- Poppler (for PDF to image conversion)
- Twilio account (for SMS functionality)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kissancare.git
cd kissancare
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils
```

For Windows:
- Download and install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
- Download and install Poppler from: http://blog.alivate.com.au/poppler-windows/

5. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
```

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

### Basic Mode
1. Click on "Basic Mode" in the navigation
2. Enter your phone number
3. Upload a clear image of the affected plant
4. Click "Analyze"
5. View the results and receive an SMS with the diagnosis

### Advanced Mode
1. Click on "Advanced Mode" in the navigation
2. Enter your phone number
3. Upload a plant image
4. (Optional) Upload a nutrient analysis PDF
5. Click "Analyze"
6. View detailed results and receive a comprehensive SMS report

## Project Structure

```
kissancare/
├── app.py              # Main application file
├── train.py           # Model training script
├── requirements.txt   # Python dependencies
├── templates/         # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── basic_mode.html
│   └── advanced_mode.html
├── static/           # Static files (CSS, JS, images)
├── uploads/          # Temporary storage for uploaded files
├── logs/            # Training logs and model checkpoints
└── processed_data/  # Processed dataset
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PlantVillage dataset for training data
- TensorFlow for machine learning capabilities
- Twilio for SMS functionality 