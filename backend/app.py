"""
Medical AI Diagnosis Support System - Flask Backend
=====================================================
Main Flask application providing REST API endpoints for:
- Medical image upload and classification
- Grad-CAM explainability visualization
- Disease information retrieval
- Hospital recommendation system

Author: Medical AI Research Team
Version: 1.0.0
"""

import os
import sys
import uuid
import json
import time
import warnings
from datetime import datetime

# Suppress warnings early  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils import (
    get_disease_info,
    get_gradcam_explanation,
    get_nearby_hospitals,
    get_health_suggestions
)

# ─── App Configuration ────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates"),
    static_folder=os.path.join(BASE_DIR, "frontend", "static")
)

CORS(app)

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "gradcam_outputs")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ─── Lazy Predictor Loading ──────────────────────────────────────────────────

_predictor = None

def _get_predictor():
    """Lazily initialize the predictor on first use."""
    global _predictor
    if _predictor is None:
        from backend.predict import get_predictor
        _predictor = get_predictor()
    return _predictor

# ─── History Storage ──────────────────────────────────────────────────────────

prediction_history = []

# ─── Helper Functions ─────────────────────────────────────────────────────────

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes: Pages ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


# ─── Routes: API Endpoints ────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Process uploaded medical image and return prediction results.
    
    Expects: multipart/form-data with 'image' file
    Returns: JSON with prediction, confidence, disease info, gradcam path
    """
    start_time = time.time()
    
    # Validate file upload
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get predictor (lazy load)
        predictor = _get_predictor()
        
        # Run prediction
        result = predictor.predict(filepath)
        
        # Generate Grad-CAM
        gradcam_filename = f"gradcam_{file_id}.png"
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
        predictor.generate_gradcam(filepath, gradcam_path)
        
        # Get disease information
        disease_info = get_disease_info(result["predicted_class"])
        
        # Get Grad-CAM explanation
        gradcam_explanation = get_gradcam_explanation(result["predicted_class"])
        
        # Get health suggestions
        health_suggestions = get_health_suggestions(
            result["predicted_class"],
            result["confidence"]
        )
        
        # Get hospital recommendations
        user_lat = request.form.get('latitude', 28.6139, type=float)
        user_lng = request.form.get('longitude', 77.2090, type=float)
        
        hospitals = get_nearby_hospitals(
            user_lat, user_lng,
            result["predicted_class"],
            max_results=5
        )
        
        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        
        # Build response
        response = {
            "success": True,
            "prediction": {
                "disease": result["predicted_class"],
                "disease_title": disease_info["title"],
                "confidence": round(result["confidence"] * 100, 2),
                "risk_level": result["risk_level"],
                "all_probabilities": {
                    k: round(v * 100, 2) for k, v in result["all_probabilities"].items()
                }
            },
            "disease_info": {
                "description": disease_info["description"],
                "icon": disease_info["icon"],
                "causes": disease_info["causes"],
                "symptoms": disease_info["symptoms"],
                "risk_factors": disease_info["risk_factors"],
                "prevention": disease_info["prevention"],
                "precautions": disease_info["precautions"],
                "urgency": disease_info["urgency"],
                "specialist": disease_info["specialist"]
            },
            "explainability": {
                "gradcam_url": f"/api/gradcam/{gradcam_filename}",
                "gradcam_overlay_url": f"/api/gradcam/{gradcam_filename.replace('.png', '_overlay.png')}",
                "explanation": gradcam_explanation
            },
            "health_suggestions": health_suggestions,
            "hospitals": hospitals,
            "metadata": {
                "file_id": file_id,
                "original_filename": file.filename,
                "image_url": f"/api/uploads/{filename}",
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_version": "MobileNetV2-v1.0"
            }
        }
        
        # Store in history
        prediction_history.append({
            "id": file_id,
            "filename": file.filename,
            "disease": result["predicted_class"],
            "confidence": round(result["confidence"] * 100, 2),
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/gradcam/<filename>')
def serve_gradcam(filename):
    """Serve Grad-CAM visualization images."""
    return send_from_directory(GRADCAM_FOLDER, filename)


@app.route('/api/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded images."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/history')
def get_history():
    """Get prediction history."""
    return jsonify({
        "success": True,
        "history": list(reversed(prediction_history)),
        "total": len(prediction_history)
    })


@app.route('/api/disease-info/<disease_type>')
def disease_info(disease_type):
    """Get detailed disease information."""
    info = get_disease_info(disease_type)
    suggestions = get_health_suggestions(disease_type, 1.0)
    return jsonify({
        "success": True,
        "disease_info": info,
        "health_suggestions": suggestions
    })


@app.route('/api/hospitals')
def hospitals():
    """Get nearby hospital recommendations."""
    lat = request.args.get('lat', 28.6139, type=float)
    lng = request.args.get('lng', 77.2090, type=float)
    disease = request.args.get('disease', 'normal')
    
    hospital_list = get_nearby_hospitals(lat, lng, disease, max_results=5)
    return jsonify({
        "success": True,
        "hospitals": hospital_list,
        "user_location": {"lat": lat, "lng": lng}
    })


@app.route('/api/system-info')
def system_info():
    """Get system information and status."""
    return jsonify({
        "success": True,
        "system": {
            "name": "Medical AI Diagnosis Support System",
            "version": "1.0.0",
            "model": "MobileNetV2 (Transfer Learning)",
            "categories": ["Normal", "Pneumonia", "Tumor", "Fracture"],
            "total_predictions": len(prediction_history),
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
    })


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  🏥 MEDICAL AI DIAGNOSIS SUPPORT SYSTEM")
    print("  📡 Starting Flask Server...")
    print("=" * 60)
    print(f"\n  🌐 Dashboard: http://localhost:5000")
    print(f"  📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"  🧠 Model path: {os.path.join(BASE_DIR, 'models', 'model.h5')}")
    print("\n" + "=" * 60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        use_reloader=False
    )
