"""
Parkinson's Disease Detection - Flask Backend (SVM COMPATIBLE)
Complete REST API for ML Model Predictions
Fixed: SVM predict_proba issue
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import joblib
import os
from datetime import datetime
import json
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# Initialize Flask App
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ==================== Model Loading ====================
MODEL_PATH = 'models/parkinsons_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

# ALL 23 FEATURES (including status for reference)
ALL_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'status', 'RPDE', 'DFA', 'spread1', 'spread2',
    'D2', 'PPE'
]

# PREDICTION FEATURES (22 features - EXCLUDES 'status')
PREDICTION_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2',
    'D2', 'PPE'
]

# Load pre-trained model and scaler
model = None
scaler = None
model_type = None

def load_models():
    """Load model and scaler with error handling"""
    global model, scaler, model_type
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            model_type = type(model).__name__
            print(f"✓ Model loaded: {model_type}")
        else:
            print("⚠ Model file not found.")
            model = None
        
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print("✓ Scaler loaded successfully")
        else:
            print("⚠ Scaler file not found.")
            scaler = None
    except Exception as e:
        print(f"✗ Error loading model/scaler: {e}")
        model = None
        scaler = None

# Load models on startup
load_models()

# ==================== Helper Functions ====================

def validate_feature_value(feature_name, value):
    """Validate individual feature values"""
    try:
        num_value = float(value)
        if np.isnan(num_value) or np.isinf(num_value):
            return False, f"{feature_name} has invalid value (NaN or Inf)"
        return True, num_value
    except (ValueError, TypeError):
        return False, f"{feature_name} must be a valid number"

def get_probabilities(model, features_scaled):
    """
    Get probability predictions from model
    Handles both sklearn models with predict_proba and those without
    """
    try:
        # Try to get predict_proba (for RandomForest, LogisticRegression, etc.)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            return probabilities
    except AttributeError:
        pass
    
    # Fallback: Use decision_function for SVM and convert to probabilities
    try:
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(features_scaled)[0]
            # Convert decision function to probability-like score (0-1 range)
            # Using sigmoid function: 1 / (1 + exp(-x))
            probability = 1.0 / (1.0 + np.exp(-decision))
            # Return as [prob_class_0, prob_class_1]
            return np.array([1 - probability, probability])
    except Exception as e:
        print(f"Warning: Could not get decision_function: {e}")
    
    # Last resort: Just use the prediction (no probability)
    prediction = model.predict(features_scaled)[0]
    return np.array([1 - prediction, prediction])

# ==================== Routes ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict Parkinson's disease from voice features
    Handles both RandomForest and SVM models
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'message': 'Please provide feature values for prediction'
            }), 400
        
        # Validate and extract ALL features
        all_features = []
        status_value = None
        missing_features = []
        invalid_features = []
        
        for feature in ALL_FEATURES:
            if feature not in data:
                missing_features.append(feature)
            else:
                is_valid, result = validate_feature_value(feature, data[feature])
                if is_valid:
                    all_features.append(result)
                    if feature == 'status':
                        status_value = int(result)
                else:
                    invalid_features.append(result)
        
        # Check for errors
        if missing_features:
            return jsonify({
                'success': False,
                'error': 'Missing features',
                'missing_count': len(missing_features),
                'missing_features': missing_features[:5],
                'message': f'Missing {len(missing_features)} required features'
            }), 400
        
        if invalid_features:
            return jsonify({
                'success': False,
                'error': 'Invalid feature values',
                'invalid_count': len(invalid_features),
                'invalid_details': invalid_features[:3],
                'message': 'Some features have invalid values'
            }), 400
        
        # Extract ONLY the 22 prediction features (exclude 'status')
        prediction_features = []
        for feature in PREDICTION_FEATURES:
            feature_index = ALL_FEATURES.index(feature)
            prediction_features.append(all_features[feature_index])
        
        # Convert to numpy array with shape (1, 22)
        features_array = np.array([prediction_features])
        
        # Scale features
        if scaler:
            try:
                import pandas as pd
                features_df = pd.DataFrame(features_array, columns=PREDICTION_FEATURES)
                features_scaled = scaler.transform(features_df)
            except Exception as scale_error:
                print(f"Scaling error: {scale_error}")
                try:
                    features_scaled = scaler.transform(features_array)
                except:
                    features_scaled = features_array
        else:
            features_scaled = features_array
        
        # Make prediction
        if model:
            try:
                print(f"Model type: {model_type}")
                print(f"Input shape: {features_scaled.shape}")
                
                # Get prediction
                prediction = model.predict(features_scaled)[0]
                
                # Get probabilities (handles both RandomForest and SVM)
                probabilities = get_probabilities(model, features_scaled)
                
                if len(probabilities) >= 2:
                    prob_healthy = float(probabilities[0])
                    prob_parkinsons = float(probabilities[1])
                else:
                    prob_healthy = 1.0 if prediction == 0 else 0.0
                    prob_parkinsons = 0.0 if prediction == 0 else 1.0
                
                # Calculate confidence (max probability)
                confidence = float(max(prob_healthy, prob_parkinsons)) * 100
                
                result = {
                    'success': True,
                    'prediction': int(prediction),
                    'status': 'Healthy' if prediction == 0 else 'Parkinson\'s Disease Detected',
                    'diagnosis': 'Healthy' if prediction == 0 else 'Possible Parkinson\'s Disease',
                    'probability_healthy': round(prob_healthy * 100, 2),
                    'probability_parkinsons': round(prob_parkinsons * 100, 2),
                    'confidence': round(confidence, 2),
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    'ground_truth': status_value,
                    'warning': '⚠️ DISCLAIMER: This is a prediction model for educational purposes only. Results should NOT be used for medical diagnosis. Always consult a qualified healthcare professional.'
                }
                
                print(f"✓ Prediction successful: {result['diagnosis']}")
                return jsonify(result), 200
                
            except Exception as pred_error:
                print(f"❌ Prediction error: {pred_error}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': 'Prediction failed',
                    'message': str(pred_error),
                    'model_type': model_type
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Model not loaded',
                'message': 'The prediction model is not available'
            }), 503
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Return list of required features and their descriptions"""
    feature_info = {
        'MDVP:Fo(Hz)': 'Average vocal fundamental frequency (Hz)',
        'MDVP:Fhi(Hz)': 'Maximum vocal fundamental frequency (Hz)',
        'MDVP:Flo(Hz)': 'Minimum vocal fundamental frequency (Hz)',
        'MDVP:Jitter(%)': 'Variation in fundamental frequency (%)',
        'MDVP:Jitter(Abs)': 'Variation in fundamental frequency (absolute)',
        'MDVP:RAP': 'Relative amplitude perturbation',
        'MDVP:PPQ': 'Pitch perturbation quotient',
        'Jitter:DDP': 'Differential jitter',
        'MDVP:Shimmer': 'Variation in amplitude (%)',
        'MDVP:Shimmer(dB)': 'Variation in amplitude (dB)',
        'Shimmer:APQ3': 'Amplitude perturbation quotient (3)',
        'Shimmer:APQ5': 'Amplitude perturbation quotient (5)',
        'MDVP:APQ': 'Amplitude perturbation quotient',
        'Shimmer:DDA': 'Differential amplitude perturbation',
        'NHR': 'Noise-to-harmonics ratio',
        'HNR': 'Harmonics-to-noise ratio',
        'status': 'Health status (0=healthy, 1=PD) - OPTIONAL/REFERENCE ONLY',
        'RPDE': 'Recurrence period density entropy',
        'DFA': 'Detrended fluctuation analysis',
        'spread1': 'Nonlinear measure spread1',
        'spread2': 'Nonlinear measure spread2',
        'D2': 'Correlation dimension',
        'PPE': 'Pitch period entropy'
    }
    
    return jsonify({
        'success': True,
        'total_features_input': len(ALL_FEATURES),
        'prediction_features_used': len(PREDICTION_FEATURES),
        'all_features': ALL_FEATURES,
        'prediction_features': PREDICTION_FEATURES,
        'descriptions': feature_info,
        'note': 'Status is optional (used for comparing with ground truth). The model uses 22 features for prediction.'
    }), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'API is running',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'model_type': model_type if model else 'N/A',
        'total_input_features': len(ALL_FEATURES),
        'prediction_features': len(PREDICTION_FEATURES),
        'model_n_features': model.n_features_in_ if model and hasattr(model, 'n_features_in_') else 'N/A',
        'has_predict_proba': model.predict_proba if model and hasattr(model, 'predict_proba') else False,
        'has_decision_function': model.decision_function if model and hasattr(model, 'decision_function') else False,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.3 (SVM Compatible)'
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Get application information"""
    return jsonify({
        'name': 'Parkinson\'s Disease Detection System',
        'version': '1.0.3 (SVM Compatible)',
        'description': 'ML-powered web application for Parkinson\'s disease detection',
        'model_type': model_type if model else 'Not loaded',
        'total_input_features': len(ALL_FEATURES),
        'prediction_features_used': len(PREDICTION_FEATURES),
        'supports_models': ['RandomForest', 'SVM', 'LogisticRegression', 'Any sklearn classifier']
    }), 200

@app.route('/api/validate', methods=['POST'])
def validate_input():
    """Validate input features before prediction"""
    try:
        data = request.get_json()
        errors = []
        warnings_list = []
        
        for feature in ALL_FEATURES:
            if feature not in data:
                errors.append(f"Missing: {feature}")
            else:
                is_valid, result = validate_feature_value(feature, data[feature])
                if not is_valid:
                    errors.append(result)
        
        return jsonify({
            'success': len(errors) == 0,
            'valid': len(errors) == 0,
            'total_features_checked': len(ALL_FEATURES),
            'errors': errors,
            'warnings': warnings_list,
            'message': 'All features valid' if len(errors) == 0 else f'{len(errors)} errors found'
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Validation failed'
        }), 500

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': 'The requested resource does not exist',
        'status_code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status_code': 500
    }), 500

# ==================== Main ====================

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("\n" + "="*70)
    print("Parkinson's Disease Detection System - SVM COMPATIBLE VERSION")
    print("="*70)
    print(f"Model Status: {'✓ Loaded' if model else '✗ Not found'}")
    if model:
        print(f"Model Type: {model_type}")
        print(f"Has predict_proba: {hasattr(model, 'predict_proba')}")
        print(f"Has decision_function: {hasattr(model, 'decision_function')}")
    print(f"Scaler Status: {'✓ Loaded' if scaler else '✗ Not found'}")
    print(f"Total Input Features: {len(ALL_FEATURES)}")
    print(f"Prediction Features: {len(PREDICTION_FEATURES)}")
    print("="*70)
    print("✓ Supports both RandomForest and SVM models")
    print("="*70 + "\n")
    
    try:
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=True
        )
    except Exception as e:
        print(f"Error starting Flask app: {e}")
