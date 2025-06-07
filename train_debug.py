# train_debug.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
import json
data_dir = 'processed_data'
image_size = 224
batch_size = 16
num_epochs = 30
initial_lr = 1e-4
val_split = 0.2
confidence_threshold = 0.8

# Create logs directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs_debug", timestamp)
os.makedirs(log_dir, exist_ok=True)

# --- LOAD DATASET ---
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset='training',
    seed=123,
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset='validation',
    seed=123,
    image_size=(image_size, image_size),
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# --- VISUALIZE A BATCH ---
for images, labels in train_ds.take(1):
    plt.figure(figsize=(12, 8))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(labels[i].numpy())])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "batch_visualization.png"))
    plt.close()

# --- AUGMENTATION ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
])

def preprocess(ds, is_train):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if is_train:
        ds = ds.map(lambda x, y: (data_augmentation(x), y))
        ds = ds.shuffle(1000)
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = preprocess(train_ds, True)
val_ds = preprocess(val_ds, False)

# --- BUILD MODEL ---
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(image_size, image_size, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Save model summary to a file
with open(os.path.join(log_dir, 'model_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# --- CLASS WEIGHTS ---
class_counts = np.zeros(num_classes)
total_samples = 0
for _, labels in train_ds:
    class_counts += np.sum(labels.numpy(), axis=0)
    total_samples += labels.shape[0]
class_weights = {
    i: total_samples / (num_classes * class_counts[i])
    for i in range(num_classes)
}

# --- TRAIN ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(os.path.join(log_dir, "best_model.h5"),
                                        save_best_only=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3),
    ]
)

# --- SAVE FINAL MODEL ---
model.save(os.path.join(log_dir, "final_model.h5"))

# --- PLOT METRICS ---
def plot_history(hist):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='train acc')
    plt.plot(hist.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_metrics.png"))
    plt.close()

plot_history(history)

# --- SAVE METADATA ---
def convert_to_serializable(obj):
    """Recursively converts objects to JSON serializable formats."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    elif hasattr(obj, 'numpy'): # Handles TensorFlow tensors
        try:
            return float(obj.numpy())
        except:
            # Fallback for more complex tensors/numpy arrays if needed
            # Attempt to convert to list, otherwise string
            np_array = obj.numpy()
            return np_array.tolist() if hasattr(np_array, 'tolist') else str(np_array)
    elif isinstance(obj, np.generic):
        return float(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Attempt to convert other unknown types to string or repr
        try:
            return str(obj)
        except:
            return repr(obj)

metadata = {
    'class_names': class_names,
    'image_size': image_size,
    'num_classes': num_classes,
    'confidence_threshold': confidence_threshold,
    'log_dir': log_dir,
    'history': convert_to_serializable(history.history),
    'class_weights': {str(k): float(v) for k, v in class_weights.items()} # Add class weights
}

with open(os.path.join(log_dir, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Debug training complete. Best and final models are saved.")
