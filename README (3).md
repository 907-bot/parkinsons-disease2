# Parkinson's Disease Detection System

A complete end-to-end machine learning web application for Parkinson's Disease detection using voice features analysis.

## 📋 Project Overview

This application provides an AI-powered diagnostic tool that analyzes voice characteristics to detect potential Parkinson's Disease. It combines:

- **Frontend**: Modern HTML5, CSS3, and JavaScript responsive interface
- **Backend**: Python Flask REST API for ML predictions
- **Machine Learning**: Pre-trained classification model (Random Forest/SVM)
- **Features**: 23 voice feature parameters for comprehensive analysis

## 🎯 Key Features

### Frontend Features
- ✨ Responsive, modern UI with intuitive navigation
- 📊 Three-tab feature input system (Basic, Advanced, Nonlinear)
- 🎨 Real-time probability visualization
- 📱 Mobile-friendly design
- 🔔 Smart notification system
- 💾 Auto-save form data (localStorage)
- ⌨️ Keyboard shortcuts (Ctrl+P for predict, Ctrl+L for sample)
- 📥 Export results as JSON
- 🌗 Professional color scheme and animations

### Backend Features
- 🔐 Secure REST API with CORS support
- 📝 Input validation and error handling
- 🎯 Real-time predictions with confidence scores
- 📊 Probability distribution analysis
- 🔍 Feature information endpoint
- ❤️ Health check endpoint
- 🚀 Scalable Flask architecture

### Medical Features
- 🎤 Comprehensive voice feature analysis
- 📈 Multiple feature categories:
  - Frequency Features (Fo, Fhi, Flo)
  - Jitter Features (perturbation in frequency)
  - Shimmer Features (perturbation in amplitude)
  - Nonlinear Measures (RPDE, DFA, D2, PPE)
  - Harmonic Features (HNR, NHR)

## 📁 Project Structure

```
parkinsons-detection/
├── app.py                          # Flask backend application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   ├── parkinsons_model.pkl       # Pre-trained ML model
│   └── scaler.pkl                 # Feature scaler
├── templates/
│   └── index.html                 # Main HTML template
├── static/
│   ├── style.css                  # Styling
│   └── script.js                  # Frontend logic
└── data/
    └── parkinsons_data.csv        # Original dataset (optional)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser

### Step 1: Clone Repository
```bash
git clone https://github.com/907-bot/parkinson-disease.git
cd parkinson-disease
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train/Prepare Model (if needed)
```bash
python train_model.py
```

### Step 5: Run Flask Application
```bash
python app.py
```

The application will start at `http://localhost:5000`

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```
Flask==2.3.0
Flask-CORS==4.0.0
numpy==1.24.0
scikit-learn==1.3.0
joblib==1.3.0
pandas==2.0.0
```

Install with: `pip install -r requirements.txt`

## 💻 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "success": true,
  "status": "API is running",
  "model_loaded": true,
  "scaler_loaded": true,
  "timestamp": "2026-01-04T12:00:00.000000"
}
```

#### 2. Get Features Information
```http
GET /api/features
```
**Response:**
```json
{
  "success": true,
  "total_features": 23,
  "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", ...],
  "descriptions": {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    ...
  }
}
```

#### 3. Make Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "MDVP:Fo(Hz)": 119.992,
  "MDVP:Fhi(Hz)": 157.302,
  "MDVP:Flo(Hz)": 74.997,
  ...
  "PPE": 0.290
}
```

**Response:**
```json
{
  "success": true,
  "prediction": 1,
  "status": "Parkinson's Disease Detected",
  "probability_healthy": 0.15,
  "probability_parkinsons": 0.85,
  "confidence": 85.0,
  "timestamp": "2026-01-04T12:00:00.000000",
  "warning": "This is a prediction model. Please consult a healthcare professional."
}
```

#### 4. Application Info
```http
GET /api/info
```
**Response:**
```json
{
  "name": "Parkinson's Disease Detection System",
  "version": "1.0.0",
  "description": "ML-powered web application...",
  "features": 23,
  "model_type": "Random Forest Classifier",
  "accuracy": "95.5%"
}
```

## 🎮 Usage Guide

### Quick Start
1. Open `http://localhost:5000` in your browser
2. Click "Load Sample" to populate sample data
3. Click "Predict" to get diagnosis
4. Review results with confidence scores

### Using Your Own Data
1. Enter voice feature values in the form fields
2. Features can be loaded from:
   - Direct input (manual entry)
   - CSV/Excel import
   - Sample data (button provided)
3. All 23 features must be provided
4. Click "Predict" for analysis

### Keyboard Shortcuts
- **Ctrl/Cmd + P**: Make prediction
- **Ctrl/Cmd + L**: Load sample data
- **Tab**: Move between fields
- **Enter**: Submit form

## 📊 Voice Features Explanation

### Frequency Features
- **MDVP:Fo(Hz)**: Average fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum fundamental frequency
- **MDVP:Flo(Hz)**: Minimum fundamental frequency

### Perturbation Features
- **Jitter**: Variation in fundamental frequency (5 measures)
- **Shimmer**: Variation in amplitude (5 measures)

### Nonlinear Features
- **RPDE**: Recurrence period density entropy
- **DFA**: Detrended fluctuation analysis
- **D2**: Correlation dimension
- **PPE**: Pitch period entropy

### Harmonic Features
- **HNR**: Harmonics-to-noise ratio
- **NHR**: Noise-to-harmonics ratio

## 🏥 Medical Disclaimer

⚠️ **IMPORTANT**
This application is designed for **educational and research purposes only**. 

- It is NOT a substitute for professional medical advice
- Results should NOT be used for self-diagnosis
- Always consult a qualified healthcare professional (neurologist)
- The model's predictions are probabilistic estimates
- Early detection and professional diagnosis are crucial

## 🧠 Model Information

### Training Data
- **Source**: UCI Machine Learning Repository
- **Samples**: 195 voice samples (147 PD, 48 healthy)
- **Features**: 23 voice parameters
- **Training/Test Split**: 70/30

### Model Performance
- **Accuracy**: ~95.5%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95%
- **Algorithm**: Random Forest Classifier (100 trees)

### Feature Importance (Top 10)
1. PPE (Pitch Period Entropy)
2. D2 (Correlation Dimension)
3. DFA (Detrended Fluctuation Analysis)
4. spread1 (Nonlinear spread measure)
5. RPDE (Recurrence Period Density Entropy)
6. HNR (Harmonics-to-Noise Ratio)
7. MDVP:Fo(Hz) (Fundamental Frequency)
8. Shimmer:DDA
9. NHR (Noise-to-Harmonics Ratio)
10. MDVP:Shimmer(dB)

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Change port in app.py
app.run(port=5001)  # Use different port
```

### Model Not Found
- Ensure `models/parkinsons_model.pkl` exists
- Run `train_model.py` if missing
- Check file permissions

### CORS Errors
- Flask-CORS is configured in `app.py`
- Ensure `CORS(app)` is enabled

### Form Not Saving
- Check localStorage is enabled in browser
- Clear browser cache if issues persist

## 📱 Deployment

### Heroku
```bash
pip freeze > requirements.txt
git push heroku main
```

### AWS/Azure/GCP
- Use similar Flask deployment procedures
- Ensure `DEBUG=False` in production
- Set environment variables for security

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## 📚 References

1. **UCI ML Repository**: https://archive.ics.uci.edu/ml/datasets/parkinsons
2. **Feature Documentation**: "Dysphonia measures and models for dysarthric speech"
3. **ML Techniques**: Scikit-learn documentation
4. **Web Framework**: Flask official documentation

## 📄 License

MIT License - see LICENSE file for details

## 👥 Author

**Abhishek Adari** (907-bot)
- GitHub: https://github.com/907-bot
- AI/ML Engineer, focusing on Healthcare AI

## 📧 Contact & Support

- **Issues**: GitHub Issues
- **Email**: [Your Email]
- **Documentation**: Check wiki and docs/

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset
- Open-source community (Flask, scikit-learn, etc.)
- Medical research community for feature definitions

---

**Last Updated**: January 4, 2026
**Version**: 1.0.0