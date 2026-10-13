"""
Medical Image Classification Model Training
=============================================
Trains a CNN using MobileNetV2 Transfer Learning for multi-class
medical image classification (Normal, Pneumonia, Tumor, Fracture).

Architecture: MobileNetV2 (pretrained on ImageNet) + Custom classifier head
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "models", "training_results")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 4
CATEGORIES = ["fracture", "normal", "pneumonia", "tumor"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Data Preparation ─────────────────────────────────────────────────────────

def create_data_generators():
    """Create training and validation data generators with augmentation."""
    
    print("\n📊 Creating Data Generators...")
    print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation - only rescale
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if not os.path.exists(train_dir):
        print(f"\n❌ Training data not found at: {train_dir}")
        print("   Run 'python dataset/download_dataset.py' first!")
        sys.exit(1)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    print(f"\n   ✅ Training samples: {train_generator.samples}")
    print(f"   ✅ Test samples: {test_generator.samples}")
    print(f"   ✅ Classes: {train_generator.class_indices}")
    
    # Save class mapping
    class_map = {v: k for k, v in train_generator.class_indices.items()}
    with open(os.path.join(MODEL_DIR, "class_mapping.json"), "w") as f:
        json.dump(class_map, f, indent=2)
    
    return train_generator, test_generator


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model():
    """Build MobileNetV2-based transfer learning model."""
    
    print("\n🧠 Building Model Architecture...")
    print("   Base: MobileNetV2 (ImageNet pretrained)")
    
    # Load MobileNetV2 without top layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\n   📐 Total Parameters: {total_params:,}")
    print(f"   🔧 Trainable Parameters: {trainable_params:,}")
    print(f"   ❄️  Non-trainable Parameters: {total_params - trainable_params:,}")
    
    return model, base_model


# ─── Fine-Tuning ──────────────────────────────────────────────────────────────

def fine_tune_model(model, base_model, train_gen, test_gen, initial_history):
    """Fine-tune the last layers of the base model."""
    
    print("\n🔬 Starting Fine-Tuning Phase...")
    
    # Unfreeze the last 30 layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "medical_model_finetuned.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    fine_tune_epochs = 10
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS + fine_tune_epochs,
        initial_epoch=len(initial_history.history['loss']),
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    """Main training pipeline."""
    
    print("\n" + "=" * 60)
    print("  🏥 MEDICAL IMAGE CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Data
    train_gen, test_gen = create_data_generators()
    
    # Step 2: Build model
    model, base_model = build_model()
    
    # Step 3: Initial training (frozen base)
    print("\n🚀 Phase 1: Training Classification Head...")
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, "medical_model_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 4: Fine-tune
    print("\n🔬 Phase 2: Fine-Tuning Base Model...")
    ft_history = fine_tune_model(model, base_model, train_gen, test_gen, history)
    
    # Step 5: Save final model
    model_path = os.path.join(MODEL_DIR, "model.h5")
    model.save(model_path)
    print(f"\n   💾 Model saved: {model_path}")
    
    # Step 6: Evaluate
    print("\n📈 Final Evaluation:")
    results = model.evaluate(test_gen, verbose=0)
    print(f"   Test Loss: {results[0]:.4f}")
    print(f"   Test Accuracy: {results[1]*100:.2f}%")
    
    # Step 7: Plot training history
    plot_training_history(history, ft_history)
    
    # Step 8: Save training metadata
    metadata = {
        "model_architecture": "MobileNetV2 + Custom Head",
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "categories": CATEGORIES,
        "epochs_phase1": EPOCHS,
        "final_accuracy": float(results[1]),
        "final_loss": float(results[0]),
    }
    
    with open(os.path.join(MODEL_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  📊 Final Accuracy: {results[1]*100:.2f}%")
    print(f"  💾 Model saved at: {model_path}")
    print("=" * 60 + "\n")
    
    return model


# ─── Visualization ─────────────────────────────────────────────────────────────

def plot_training_history(history, ft_history=None):
    """Plot and save training history charts."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#FF5722')
    if ft_history:
        offset = len(history.history['accuracy'])
        ft_epochs = range(offset, offset + len(ft_history.history['accuracy']))
        axes[0].plot(ft_epochs, ft_history.history['accuracy'], '--', label='FT Train Acc', color='#4CAF50')
        axes[0].plot(ft_epochs, ft_history.history['val_accuracy'], '--', label='FT Val Acc', color='#FF9800')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='#2196F3')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#FF5722')
    if ft_history:
        axes[1].plot(ft_epochs, ft_history.history['loss'], '--', label='FT Train Loss', color='#4CAF50')
        axes[1].plot(ft_epochs, ft_history.history['val_loss'], '--', label='FT Val Loss', color='#FF9800')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   📊 Training plots saved to {RESULTS_DIR}/training_history.png")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🖥️  GPU detected: {len(gpus)} device(s)")
    else:
        print("🖥️  Running on CPU")
    
    train()
