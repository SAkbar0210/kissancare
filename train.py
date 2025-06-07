import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import json
import logging
import gc
import psutil
import time
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def reorganize_data_structure():
    """
    Reorganize data from train/test splits to class-based structure
    """
    base_dir = 'processed_data'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.info("Data structure already correct or directories not found")
        return False
    
    logger.info("Reorganizing data structure...")
    
    # Create backup
    backup_dir = f"{base_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(base_dir, backup_dir)
    logger.info(f"Backup created at: {backup_dir}")
    
    # Get all class names from train directory
    class_names = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
    
    # Create new structure
    new_base_dir = f"{base_dir}_reorganized"
    os.makedirs(new_base_dir, exist_ok=True)
    
    for class_name in class_names:
        class_dir = os.path.join(new_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy from train
        train_class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(train_class_dir):
            for img_file in os.listdir(train_class_dir):
                src = os.path.join(train_class_dir, img_file)
                dst = os.path.join(class_dir, f"train_{img_file}")
                shutil.copy2(src, dst)
        
        # Copy from test
        test_class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(test_class_dir):
            for img_file in os.listdir(test_class_dir):
                src = os.path.join(test_class_dir, img_file)
                dst = os.path.join(class_dir, f"test_{img_file}")
                shutil.copy2(src, dst)
        
        logger.info(f"Reorganized class: {class_name}")
    
    # Replace old structure
    shutil.rmtree(base_dir)
    shutil.move(new_base_dir, base_dir)
    
    logger.info("Data reorganization completed!")
    return True

# Configure GPU memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")
        
        # Log GPU info
        for gpu in gpus:
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu)
                logger.info(f"GPU {gpu.name} memory: {memory_info['current'] / 1024**2:.2f} MB")
            except:
                logger.info(f"GPU {gpu.name} detected")
except Exception as e:
    logger.warning(f"Could not configure GPU memory growth: {str(e)}")

# Enable mixed precision for RTX 4050
if tf.config.list_physical_devices('GPU'):
    try:
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision enabled for better performance")
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {str(e)}")

# Optimized config for RTX 4050 (4GB VRAM)
class Config:
    def __init__(self):
        self.data_dir = 'processed_data'
        self.image_size = 224  # Good balance for RTX 4050
        self.batch_size = 16   # Optimized for 4GB VRAM
        self.epochs = 50  # Increased epochs
        self.initial_lr = 1e-4  # Further reduced initial learning rate
        self.dropout_rate = 0.3
        self.confidence_threshold = 0.8
        self.early_stopping_patience = 10  # Increased patience
        self.validation_split = 0.2
        
        # RTX 4050 optimizations
        self.prefetch_buffer = tf.data.AUTOTUNE
        self.num_parallel_calls = tf.data.AUTOTUNE

config = Config()

# Check and reorganize data if needed
data_reorganized = reorganize_data_structure()

# Create logs directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", timestamp)
os.makedirs(log_dir, exist_ok=True)

logger.info("Setting up data pipeline...")
log_memory_usage()

def create_dataset(data_dir, validation_split=0.2, subset=None):
    """Create tf.data dataset with proper error handling"""
    try:
        if subset == 'training':
            dataset = keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset='training',
                seed=123,
                image_size=(config.image_size, config.image_size),
                batch_size=config.batch_size,
                label_mode='categorical'
            )
        elif subset == 'validation':
            dataset = keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset='validation',
                seed=123,
                image_size=(config.image_size, config.image_size),
                batch_size=config.batch_size,
                label_mode='categorical'
            )
        else:
            dataset = keras.utils.image_dataset_from_directory(
                data_dir,
                image_size=(config.image_size, config.image_size),
                batch_size=config.batch_size,
                label_mode='categorical'
            )
        
        return dataset
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

# Create datasets
train_dataset = create_dataset(config.data_dir, config.validation_split, 'training')
val_dataset = create_dataset(config.data_dir, config.validation_split, 'validation')

# Get class information
class_names = train_dataset.class_names
num_classes = len(class_names)
logger.info(f"Found {num_classes} classes: {class_names}")

# Optimize datasets for performance
def optimize_dataset(dataset, is_training=False):
    """Optimize dataset for RTX 4050 performance"""
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=config.num_parallel_calls
    )
    
    if is_training:
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),  # Reduced rotation
            layers.RandomZoom(0.1),      # Reduced zoom
            layers.RandomBrightness(0.1), # Added brightness
            layers.RandomContrast(0.1),   # Reduced contrast
        ])
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=config.num_parallel_calls
        )
        dataset = dataset.shuffle(1000)
    
    dataset = dataset.cache()
    dataset = dataset.prefetch(config.prefetch_buffer)
    
    return dataset

train_dataset = optimize_dataset(train_dataset, is_training=True)
val_dataset = optimize_dataset(val_dataset, is_training=False)

# Calculate class weights
logger.info("Calculating class weights...")
class_counts = np.zeros(num_classes)
total_samples = 0

for images, labels in train_dataset:
    class_counts += np.sum(labels.numpy(), axis=0)
    total_samples += labels.shape[0]

class_weights = {
    i: total_samples / (num_classes * class_counts[i]) 
    for i in range(num_classes)
}

logger.info("Class distribution:")
for i, class_name in enumerate(class_names):
    logger.info(f"{class_name}: {int(class_counts[i])} samples (weight: {class_weights[i]:.2f})")

# Clear memory
keras.backend.clear_session()
gc.collect()
log_memory_usage()

# Custom function to make objects JSON serializable
def make_json_serializable(obj):
    """Convert numpy/tensor objects to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'numpy'):
        return float(obj.numpy())
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    else:
        return obj

def build_efficient_model(input_shape, num_classes):
    """Build an efficient model optimized for RTX 4050"""
    
    # Use EfficientNetB0 - better than MobileNetV2 for plant diseases
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        drop_connect_rate=0.3  # Increased dropout for better regularization
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head optimized for plant diseases
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Increased dropout
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Increased dropout
        layers.Dense(num_classes, activation='softmax', dtype='float32')  # Ensure float32 output
    ])
    
    return model, base_model

logger.info("Building optimized model...")
model, base_model = build_efficient_model((config.image_size, config.image_size, 3), num_classes)
log_memory_usage()

# Compile model with proper optimizer settings
optimizer = keras.optimizers.Adam(
    learning_rate=config.initial_lr,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

logger.info(f"Model compiled. Total parameters: {model.count_params():,}")

# Custom callbacks with JSON serialization fix
class SafeMetricsCallback(keras.callbacks.Callback):
    """Custom callback that handles tensor serialization properly"""

    def __init__(self, validation_data, **kwargs):
        super().__init__(**kwargs)
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Create a new dictionary with converted values
            processed_logs = {}
            for key, value in logs.items():
                # Attempt to convert tensors and numpy types to float, handle others gracefully
                try:
                    if hasattr(value, 'numpy'):
                        processed_logs[key] = float(value.numpy())
                    elif isinstance(value, (np.float16, np.float32, np.float64, np.int32, np.int64)):
                        processed_logs[key] = float(value)
                    else:
                        processed_logs[key] = value # Keep non-numeric values as is
                except Exception:
                    processed_logs[key] = value # Fallback to original value on error

            # Update the original logs dictionary with the processed values
            logs.update(processed_logs)

            # Calculate confidence metrics using the potentially updated logs
            try:
                val_predictions = self.model.predict(self.validation_data, verbose=0)
                val_confidences = np.max(val_predictions, axis=1)

                high_confidence_ratio = float(np.mean(val_confidences > self.params['metrics_config']['confidence_threshold']))
                avg_confidence = float(np.mean(val_confidences))

                # Add calculated metrics to logs (they should be serializable floats now)
                logs['high_confidence_ratio'] = high_confidence_ratio
                logs['avg_confidence'] = avg_confidence

                print(f"\nEpoch {epoch + 1} - High Confidence Ratio: {high_confidence_ratio:.2%}")
                print(f"Average Confidence: {avg_confidence:.2%}")

            except Exception as e:
                logger.warning(f"Could not calculate confidence metrics: {str(e)}")

class MemoryCallback(keras.callbacks.Callback):
    """Monitor memory usage during training"""
    
    def on_epoch_begin(self, epoch, logs=None):
        log_memory_usage()
        # Clear memory
        gc.collect()
    
    def on_epoch_end(self, epoch, logs=None):
        log_memory_usage()
        gc.collect()

# Learning rate schedule for fine-tuning
def create_lr_schedule():
    def lr_schedule(epoch, lr):
        if epoch < 10:  # Start with a warmup period
            return lr
        elif epoch < 20:
            return lr * 0.5
        elif epoch < 30:
            return lr * 0.1
        else:
            return lr * 0.01
    return lr_schedule

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(log_dir, "best_model.h5"),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=config.early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.LearningRateScheduler(create_lr_schedule(), verbose=1),
    SafeMetricsCallback(validation_data=val_dataset),
    MemoryCallback()
]

logger.info("Starting training phase 1 (frozen base)...")
try:
    # Phase 1: Train with frozen base
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,  # Increased initial training phase
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    logger.info("Phase 1 completed. Starting fine-tuning phase...")
    
    # Phase 2: Fine-tune last layers
    base_model.trainable = True
    
    # Freeze early layers, unfreeze last few
    for layer in base_model.layers[:-30]:  # Unfreeze more layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.initial_lr/5),  # Less aggressive reduction
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        initial_epoch=20,
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Combine histories
    history = history1
    for key in history2.history:
        if key in history.history:
            history.history[key].extend(history2.history[key])
        else:
            history.history[key] = history2.history[key]
            
except Exception as e:
    logger.error(f"Error during training: {str(e)}")
    raise

# Save model and metadata
try:
    model_metadata = {
        'confidence_threshold': float(config.confidence_threshold),
        'class_names': [str(name) for name in class_names],
        'image_size': int(config.image_size),
        'model_type': 'EfficientNetB0',
        'num_classes': int(num_classes),
        'training_date': timestamp,
        'total_parameters': int(model.count_params()),
        'data_reorganized': data_reorganized,
        'class_weights': {int(k): v for k, v in class_weights.items()} # Keep original values for now, conversion happens later
    }

    # Use the helper function to ensure all elements are JSON serializable
    safe_model_metadata = make_json_serializable(model_metadata)

    with open(os.path.join(log_dir, 'model_metadata.json'), 'w') as f:
        json.dump(safe_model_metadata, f, indent=4)

    # Save final model
    model.save(os.path.join(log_dir, 'plant_disease_model.h5'))
    logger.info("Model and metadata saved successfully")
    
except Exception as e:
    logger.error(f"Error saving model: {str(e)}")

# Final evaluation
try:
    logger.info("Performing final evaluation...")
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    
    # Get predictions for detailed analysis
    predictions = model.predict(val_dataset)
    confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    high_confidence_ratio = float(np.mean(confidences > config.confidence_threshold))
    avg_confidence = float(np.mean(confidences))
    
    print(f"\n{'='*50}")
    print("FINAL MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"High Confidence Predictions (>{config.confidence_threshold}): {high_confidence_ratio:.2%}")
    print(f"Average Confidence: {avg_confidence:.2%}")
    
    # Class-wise analysis
    print(f"\n{'='*30}")
    print("CLASS-WISE ANALYSIS")
    print(f"{'='*30}")
    
    # Get true labels
    true_labels = []
    for _, labels in val_dataset:
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
    true_labels = np.array(true_labels)
    
    for i, class_name in enumerate(class_names):
        class_mask = true_labels == i
        if np.sum(class_mask) > 0:
            class_predictions = predictions[class_mask]
            class_confidences = np.max(class_predictions, axis=1)
            class_accuracy = np.mean(np.argmax(class_predictions, axis=1) == i)
            
            print(f"\n{class_name}:")
            print(f"  Samples: {np.sum(class_mask)}")
            print(f"  Accuracy: {class_accuracy:.2%}")
            print(f"  Avg Confidence: {np.mean(class_confidences):.2%}")
            print(f"  High Confidence Ratio: {np.mean(class_confidences > config.confidence_threshold):.2%}")
    
except Exception as e:
    logger.error(f"Error during evaluation: {str(e)}")

# Save training history safely
try:
    safe_history = make_json_serializable(history.history)
    
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(safe_history, f, indent=4)
    
    # Also save as pickle for Python use
    with open(os.path.join(log_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(safe_history, f)
    
    logger.info("Training history saved successfully")
    
except Exception as e:
    logger.error(f"Error saving history: {str(e)}")

# Create training plots
try:
    def plot_training_metrics(history_dict, save_dir):
        """Create and save training plots"""
        if not history_dict or 'accuracy' not in history_dict:
            logger.warning("No training history available for plotting")
            return
        
        plt.style.use('default')  # Use default style for compatibility
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history_dict['accuracy']) + 1)
        
        # Accuracy plot
        ax1.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history_dict:
            ax1.plot(epochs, history_dict['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history_dict:
            ax2.plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        if 'lr' in history_dict:
            ax3.plot(epochs, history_dict['lr'], 'g-', linewidth=2)
            ax3.set_title('Learning Rate', fontsize=14)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Confidence metrics plot
        if 'avg_confidence' in history_dict:
            ax4.plot(epochs, history_dict['avg_confidence'], 'purple', label='Avg Confidence', linewidth=2)
            if 'high_confidence_ratio' in history_dict:
                ax4.plot(epochs, history_dict['high_confidence_ratio'], 'orange', label='High Conf Ratio', linewidth=2)
            ax4.set_title('Confidence Metrics', fontsize=14)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Confidence')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training plots saved successfully")
    
    plot_training_metrics(safe_history, log_dir)
    
except Exception as e:
    logger.error(f"Error creating plots: {str(e)}")

# Final cleanup
keras.backend.clear_session()
gc.collect()
log_memory_usage()

print(f"\n{'='*60}")
print("🌱 PLANT DISEASE CLASSIFICATION TRAINING COMPLETED! 🌱")
print(f"{'='*60}")
print(f"📁 Models and logs saved in: {log_dir}")
print(f"🎯 Final Validation Accuracy: {val_accuracy:.2%}")
print(f"💪 Model ready for real-time inference!")
print(f"🔧 Optimized for RTX 4050 - Perfect for your setup!")

if data_reorganized:
    print(f"📊 Data structure was reorganized automatically")
    print(f"💾 Original data backed up")

print(f"{'='*60}")

logger.info("Training completed successfully!")
logger.info(f"Model saved at: {os.path.join(log_dir, 'plant_disease_model.h5')}")