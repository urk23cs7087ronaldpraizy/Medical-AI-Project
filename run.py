#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  MedVision AI — Medical Image Diagnosis Support System
  Main entry point to launch the application
═══════════════════════════════════════════════════════════════════

Usage:
    python3 run.py              # Run the web application
    python3 run.py --setup      # Setup: generate dataset + build model
    python3 run.py --train      # Train the AI model only
    python3 run.py --dataset    # Generate dataset only
"""

import os
import sys
import warnings
import argparse

# Suppress all warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def setup_all():
    """Run full setup: generate dataset and build/download model."""
    print("\n" + "=" * 60)
    print("  🏥 MedVision AI — Full Setup")
    print("=" * 60)
    
    print("\n📦 Step 1: Preparing Dataset...")
    from dataset.download_dataset import generate_dataset
    generate_dataset(num_per_class=250)
    
    print("\n🧠 Step 2: Building AI Model...")
    from backend.predict import get_predictor
    predictor = get_predictor()
    
    print("\n" + "=" * 60)
    print("  ✅ Setup Complete!")
    print("  Run 'python3 run.py' to start the web application")
    print("=" * 60 + "\n")


def train_model():
    """Train the AI model on the dataset."""
    print("\n🧠 Training AI Model...")
    from models.train_model import train
    train()


def generate_dataset():
    """Generate the synthetic dataset."""
    print("\n📦 Generating Dataset...")
    from dataset.download_dataset import generate_dataset as gen
    gen(num_per_class=250)


def run_app():
    """Launch the Flask web application."""
    print("\n" + "=" * 60)
    print("  🏥 MedVision AI — Medical Image Diagnosis System")
    print("=" * 60)
    print(f"\n  🌐 Open in browser: http://localhost:5002")
    print(f"\n  Press Ctrl+C to stop the server\n")
    
    # Import and run Flask - model loads lazily on first prediction
    from backend.app import app
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        use_reloader=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MedVision AI — Medical Image Diagnosis Support System'
    )
    parser.add_argument('--setup', action='store_true',
                       help='Run full setup (dataset + model)')
    parser.add_argument('--train', action='store_true',
                       help='Train the AI model')
    parser.add_argument('--dataset', action='store_true',
                       help='Generate dataset only')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_all()
    elif args.train:
        train_model()
    elif args.dataset:
        generate_dataset()
    else:
        run_app()
