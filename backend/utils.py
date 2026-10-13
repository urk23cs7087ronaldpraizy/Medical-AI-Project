"""
Utility Module for Medical AI System
======================================
Contains medical knowledge base, hospital recommendation,
and helper functions.
"""

import os
import json
import math
import random

# ─── Medical Knowledge Base ───────────────────────────────────────────────────

DISEASE_INFO = {
    "normal": {
        "title": "Normal / Healthy",
        "description": "The medical image appears normal with no significant abnormalities detected. The AI system has not identified any pathological findings in the analyzed image.",
        "icon": "✅",
        "causes": [
            "This is a healthy scan - no disease causes apply."
        ],
        "symptoms": [
            "No abnormal symptoms expected.",
            "If you experience any unusual symptoms, consult a healthcare professional."
        ],
        "risk_factors": [
            "Maintain regular health checkups",
            "Follow a healthy lifestyle",
            "Exercise regularly",
            "Balanced diet and adequate sleep"
        ],
        "prevention": [
            "Continue regular medical checkups",
            "Maintain a healthy and active lifestyle",
            "Stay hydrated and eat nutritious food",
            "Avoid smoking and excessive alcohol consumption",
            "Get vaccinated according to recommended schedules"
        ],
        "precautions": [
            "Continue routine health screenings",
            "Maintain vaccination schedules",
            "Regular physical exercise",
            "Balanced nutrition"
        ],
        "urgency": "none",
        "specialist": "General Physician"
    },
    
    "pneumonia": {
        "title": "Pneumonia Detected",
        "description": "The AI system has detected patterns consistent with pneumonia in the chest X-ray. Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm, fever, chills, and difficulty breathing.",
        "icon": "🫁",
        "causes": [
            "Bacterial infection (Streptococcus pneumoniae most common)",
            "Viral infection (Influenza, COVID-19, RSV)",
            "Fungal infection (in immunocompromised patients)",
            "Aspiration of food or liquid into the lungs",
            "Hospital-acquired infections (nosocomial pneumonia)",
            "Weakened immune system"
        ],
        "symptoms": [
            "Persistent cough (may produce phlegm)",
            "High fever with chills and sweating",
            "Shortness of breath and rapid breathing",
            "Sharp chest pain during deep breathing or coughing",
            "Fatigue and muscle aches",
            "Nausea, vomiting, or diarrhea",
            "Confusion (especially in older adults)",
            "Bluish color of lips or fingernails (cyanosis)"
        ],
        "risk_factors": [
            "Age (very young children and adults over 65)",
            "Chronic lung diseases (COPD, asthma)",
            "Smoking",
            "Weakened immune system (HIV/AIDS, chemotherapy)",
            "Hospitalization (especially ICU with ventilator)",
            "Diabetes, heart disease, or liver disease",
            "Recent respiratory infections"
        ],
        "prevention": [
            "Get vaccinated (pneumococcal vaccine, flu vaccine)",
            "Practice good hygiene (frequent handwashing)",
            "Don't smoke or quit smoking",
            "Keep immune system strong with proper nutrition",
            "Get adequate rest and manage stress",
            "Wear masks during respiratory illness outbreaks",
            "Avoid close contact with sick individuals"
        ],
        "precautions": [
            "Seek immediate medical attention",
            "Complete the full course of prescribed antibiotics",
            "Rest and stay hydrated",
            "Monitor oxygen levels with a pulse oximeter",
            "Isolate if infectious pneumonia is suspected",
            "Follow up with chest X-ray after treatment"
        ],
        "urgency": "high",
        "specialist": "Pulmonologist"
    },
    
    "tumor": {
        "title": "Brain Tumor Detected",
        "description": "The AI system has identified patterns consistent with a brain tumor in the MRI scan. A brain tumor is an abnormal growth of cells within the brain or central spinal canal. Tumors can be benign (non-cancerous) or malignant (cancerous). Early detection is crucial for treatment planning.",
        "icon": "🧠",
        "causes": [
            "Genetic mutations in brain cells",
            "Inherited genetic conditions (neurofibromatosis, Li-Fraumeni syndrome)",
            "Exposure to ionizing radiation",
            "Family history of brain tumors",
            "Compromised immune system",
            "Environmental factors (industrial chemicals exposure)",
            "Age-related cellular changes"
        ],
        "symptoms": [
            "Persistent headaches (new pattern or increasingly severe)",
            "Seizures (especially new-onset in adults)",
            "Vision problems (blurred, double vision, loss of peripheral vision)",
            "Gradual loss of sensation or movement in limbs",
            "Difficulty with balance and coordination",
            "Speech difficulties",
            "Personality or behavior changes",
            "Nausea and vomiting (especially morning)",
            "Memory problems and confusion",
            "Hearing problems"
        ],
        "risk_factors": [
            "Exposure to ionizing radiation",
            "Family history of brain tumors",
            "Age (certain types more common in specific age groups)",
            "Previous cancer history",
            "Genetic syndromes (neurofibromatosis, tuberous sclerosis)",
            "Compromised immune system",
            "Exposure to certain chemicals"
        ],
        "prevention": [
            "Avoid unnecessary radiation exposure",
            "Regular neurological check-ups if family history exists",
            "Maintain a healthy lifestyle and strong immune system",
            "Report persistent headaches or neurological symptoms early",
            "Genetic counseling if family history of tumors",
            "Limit exposure to environmental toxins",
            "Protect head from injuries"
        ],
        "precautions": [
            "Seek immediate neurology consultation",
            "Get comprehensive MRI evaluation",
            "Consider biopsy for definitive diagnosis",
            "Discuss treatment options (surgery, radiation, chemotherapy)",
            "Seek second opinion for treatment planning",
            "Join support groups for emotional well-being",
            "Regular follow-up scans to monitor progression"
        ],
        "urgency": "critical",
        "specialist": "Neurologist / Neurosurgeon"
    },
    
    "fracture": {
        "title": "Bone Fracture Detected",
        "description": "The AI system has detected patterns consistent with a bone fracture in the X-ray image. A fracture is a break in the continuity of bone tissue. Fractures can range from thin cracks to complete breaks and require proper medical evaluation and treatment for optimal healing.",
        "icon": "🦴",
        "causes": [
            "Trauma or injury (falls, accidents, sports injuries)",
            "Overuse and repetitive stress (stress fractures)",
            "Osteoporosis (weakened bones)",
            "Pathological conditions (bone tumors, infections)",
            "Age-related bone density loss",
            "Nutritional deficiencies (calcium, vitamin D)",
            "Certain medications (long-term corticosteroid use)"
        ],
        "symptoms": [
            "Intense pain at the injury site",
            "Swelling and bruising around the affected area",
            "Deformity or abnormal shape of the limb",
            "Difficulty bearing weight or moving the affected area",
            "Tenderness to touch",
            "Numbness or tingling below the fracture",
            "Grinding sensation (crepitus) during movement",
            "Visible bone protrusion in severe cases"
        ],
        "risk_factors": [
            "Osteoporosis or low bone density",
            "Advanced age",
            "Participation in high-impact sports",
            "Nutritional deficiencies (calcium, vitamin D)",
            "Sedentary lifestyle",
            "Smoking and excessive alcohol use",
            "Certain medical conditions (cancer, osteogenesis imperfecta)",
            "Previous fracture history"
        ],
        "prevention": [
            "Ensure adequate calcium and vitamin D intake",
            "Regular weight-bearing exercise",
            "Fall prevention measures (especially for elderly)",
            "Bone density screening for at-risk individuals",
            "Wear appropriate protective gear during sports",
            "Maintain a healthy body weight",
            "Quit smoking and limit alcohol intake",
            "Home safety modifications to prevent falls"
        ],
        "precautions": [
            "Immobilize the affected area immediately",
            "Apply ice to reduce swelling (20 min on, 20 min off)",
            "Seek emergency medical care",
            "Follow orthopedic treatment plan",
            "Physical therapy for rehabilitation",
            "Adequate nutrition for bone healing",
            "Follow-up X-rays to monitor healing progress"
        ],
        "urgency": "high",
        "specialist": "Orthopedic Surgeon"
    }
}

# ─── Grad-CAM Explanations ────────────────────────────────────────────────────

GRADCAM_EXPLANATIONS = {
    "normal": "The heatmap shows uniform distribution across the image with no focused activation areas. This indicates that the AI model did not find any specific region of concern. The scan appears normal.",
    "pneumonia": "The highlighted regions in the heatmap indicate areas of increased opacity in the lung fields. These bright areas correspond to possible fluid accumulation or inflammatory consolidation in the alveoli, which is consistent with pneumonia. The AI focuses on the lung parenchyma where abnormal tissue density was detected.",
    "tumor": "The heatmap highlights a localized area of abnormal tissue density. This focused region of high activation indicates an area where the AI detected patterns consistent with abnormal cell growth (tumor mass). The surrounding edema and mass effect may also be contributing to the detection.",
    "fracture": "The heatmap shows concentrated activation along a line or region in the bone structure. This highlighted area indicates a discontinuity in the bone cortex, consistent with a fracture line. The AI has detected the disruption in normal bone architecture at this location."
}

# ─── Hospital Recommendation System ──────────────────────────────────────────

# Default hospital database (used when no API is available)
HOSPITALS_DB = [
    {
        "name": "Apollo Hospital",
        "address": "21 Greams Lane, Off Greams Road",
        "city": "Chennai",
        "phone": "+91-44-2829-3333",
        "rating": 4.5,
        "specialties": ["Cardiology", "Neurology", "Oncology", "Orthopedics", "Pulmonology"],
        "lat": 13.0604,
        "lng": 80.2496,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "AIIMS - All India Institute of Medical Sciences",
        "address": "Sri Aurobindo Marg, Ansari Nagar",
        "city": "New Delhi",
        "phone": "+91-11-2658-8500",
        "rating": 4.8,
        "specialties": ["Neurosurgery", "Oncology", "Cardiology", "Pulmonology", "Orthopedics"],
        "lat": 28.5672,
        "lng": 77.2100,
        "emergency": True,
        "type": "Government Hospital"
    },
    {
        "name": "Fortis Memorial Research Institute",
        "address": "Sector 44, Opposite HUDA City Centre",
        "city": "Gurugram",
        "phone": "+91-124-4962-200",
        "rating": 4.6,
        "specialties": ["Oncology", "Neurology", "Orthopedics", "Pulmonology"],
        "lat": 28.4432,
        "lng": 77.0405,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "Medanta - The Medicity",
        "address": "CH Baktawar Singh Road, Sector 38",
        "city": "Gurugram",
        "phone": "+91-124-4141-414",
        "rating": 4.7,
        "specialties": ["Cardiac Sciences", "Neurosciences", "Oncology", "Orthopedics"],
        "lat": 28.4395,
        "lng": 77.0422,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "Tata Memorial Hospital",
        "address": "Dr Ernest Borges Marg, Parel",
        "city": "Mumbai",
        "phone": "+91-22-2417-7000",
        "rating": 4.7,
        "specialties": ["Oncology", "Radiation Therapy", "Surgical Oncology"],
        "lat": 19.0048,
        "lng": 72.8432,
        "emergency": True,
        "type": "Cancer Hospital"
    },
    {
        "name": "Narayana Health City",
        "address": "258/A, Bommasandra Industrial Area",
        "city": "Bengaluru",
        "phone": "+91-80-7122-2222",
        "rating": 4.5,
        "specialties": ["Cardiac Sciences", "Neurology", "Oncology", "Orthopedics"],
        "lat": 12.8123,
        "lng": 77.6740,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "CMC Vellore",
        "address": "Ida Scudder Road",
        "city": "Vellore",
        "phone": "+91-416-228-1000",
        "rating": 4.8,
        "specialties": ["Neurology", "Orthopedics", "Oncology", "Pulmonology", "General Medicine"],
        "lat": 12.9249,
        "lng": 79.1325,
        "emergency": True,
        "type": "Teaching Hospital"
    },
    {
        "name": "Max Super Speciality Hospital",
        "address": "1, 2, Press Enclave Road, Saket",
        "city": "New Delhi",
        "phone": "+91-11-2651-5050",
        "rating": 4.4,
        "specialties": ["Oncology", "Neurosciences", "Cardiac Sciences", "Orthopedics"],
        "lat": 28.5275,
        "lng": 77.2137,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "Manipal Hospital",
        "address": "98, HAL Old Airport Road",
        "city": "Bengaluru",
        "phone": "+91-80-2502-4444",
        "rating": 4.5,
        "specialties": ["Neurosurgery", "Oncology", "Orthopedics", "Critical Care"],
        "lat": 12.9592,
        "lng": 77.6474,
        "emergency": True,
        "type": "Multi-Specialty"
    },
    {
        "name": "PGIMER - Post Graduate Institute",
        "address": "Sector 12",
        "city": "Chandigarh",
        "phone": "+91-172-274-6018",
        "rating": 4.6,
        "specialties": ["Neurology", "Pulmonology", "Orthopedics", "Oncology"],
        "lat": 30.7649,
        "lng": 76.7784,
        "emergency": True,
        "type": "Government Hospital"
    }
]

# Specialist mapping for disease types
SPECIALIST_MAP = {
    "normal": ["General Medicine"],
    "pneumonia": ["Pulmonology", "General Medicine", "Critical Care"],
    "tumor": ["Oncology", "Neurology", "Neurosurgery", "Neurosciences"],
    "fracture": ["Orthopedics", "Emergency Medicine"]
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers."""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_nearby_hospitals(user_lat=28.6139, user_lng=77.2090, disease_type="normal", max_results=5):
    """
    Get recommended hospitals based on user location and disease type.
    
    Args:
        user_lat: User's latitude (default: New Delhi)
        user_lng: User's longitude (default: New Delhi)
        disease_type: Detected disease category
        max_results: Maximum number of results
    
    Returns:
        List of hospital recommendations with distance
    """
    required_specialties = SPECIALIST_MAP.get(disease_type, ["General Medicine"])
    
    hospitals_with_distance = []
    
    for hospital in HOSPITALS_DB:
        # Calculate distance
        distance = haversine_distance(
            user_lat, user_lng,
            hospital["lat"], hospital["lng"]
        )
        
        # Check specialty match
        matching_specs = set(hospital["specialties"]) & set(required_specialties)
        relevance_score = len(matching_specs) / len(required_specialties) if required_specialties else 0
        
        hospital_info = {
            "name": hospital["name"],
            "address": hospital["address"],
            "city": hospital["city"],
            "phone": hospital["phone"],
            "rating": hospital["rating"],
            "distance_km": round(distance, 1),
            "specialties": hospital["specialties"],
            "matching_specialties": list(matching_specs),
            "relevance_score": relevance_score,
            "emergency": hospital["emergency"],
            "type": hospital["type"]
        }
        
        hospitals_with_distance.append(hospital_info)
    
    # Sort by distance first, then by relevance
    hospitals_with_distance.sort(key=lambda x: (x["distance_km"], -x["relevance_score"]))
    
    return hospitals_with_distance[:max_results]


def get_disease_info(disease_type):
    """Get comprehensive disease information."""
    return DISEASE_INFO.get(disease_type, DISEASE_INFO["normal"])


def get_gradcam_explanation(disease_type):
    """Get human-readable explanation for the Grad-CAM visualization."""
    return GRADCAM_EXPLANATIONS.get(disease_type, GRADCAM_EXPLANATIONS["normal"])


def get_health_suggestions(disease_type, confidence):
    """Generate smart health suggestions based on prediction."""
    base_suggestions = []
    
    if disease_type == "normal":
        base_suggestions = [
            "✅ Continue maintaining your health with regular checkups",
            "🥗 Follow a balanced diet rich in fruits and vegetables",
            "🏃 Engage in regular physical activity (150 min/week)",
            "💤 Ensure adequate sleep (7-9 hours per night)",
            "💧 Stay hydrated (8+ glasses of water daily)"
        ]
    elif disease_type == "pneumonia":
        base_suggestions = [
            "🏥 Seek immediate medical consultation with a pulmonologist",
            "💊 Complete the full course of prescribed medications",
            "🛏️ Get adequate rest to support recovery",
            "💧 Increase fluid intake to prevent dehydration",
            "🌡️ Monitor temperature and oxygen levels regularly",
            "😷 Practice respiratory hygiene to prevent spread",
            "📅 Schedule follow-up chest X-ray after treatment"
        ]
    elif disease_type == "tumor":
        base_suggestions = [
            "🏥 Seek immediate consultation with a neurologist/neurosurgeon",
            "🔬 Request comprehensive MRI with contrast for detailed evaluation",
            "📋 Get a biopsy to determine tumor type (benign vs malignant)",
            "👨‍⚕️ Consider a multidisciplinary team approach for treatment",
            "💊 Discuss treatment options: surgery, radiation, chemotherapy",
            "🧠 Consider neuropsychological evaluation",
            "💪 Join a support group for emotional well-being",
            "📅 Plan regular follow-up imaging to monitor progression"
        ]
    elif disease_type == "fracture":
        base_suggestions = [
            "🏥 Visit an orthopedic specialist immediately",
            "❄️ Apply ice and elevate the affected area",
            "🚫 Avoid putting weight on the injured area",
            "💊 Take prescribed pain medication as directed",
            "🦴 Ensure adequate calcium and vitamin D intake",
            "🏋️ Follow physical therapy plan for rehabilitation",
            "📅 Schedule follow-up X-rays to monitor healing",
            "⚕️ Complete the full recommended immobilization period"
        ]
    
    if confidence < 0.5:
        base_suggestions.insert(0, "⚠️ Low confidence prediction - manual review by a specialist is strongly recommended")
    
    return base_suggestions
