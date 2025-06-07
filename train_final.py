# train_final.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import json

# --- CONFIGURATION ---
data_dir = 'processed_data'
image_size = 224
batch_size = 16
num_epochs = 50
initial_lr = 1e-4
val_split = 0.2

# Create logs directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs_final", timestamp)
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

# --- AUGMENTATION PIPELINE ---
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
base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(image_size, image_size, 3)
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

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

# --- TRAIN PHASE 1: FROZEN BASE ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(os.path.join(log_dir, "best_model.h5"), save_best_only=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]
)

# --- UNFREEZE LAST 80 LAYERS FOR FINE-TUNING ---
base_model.trainable = True
for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr / 5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- TRAIN PHASE 2: FINE-TUNING ---
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    initial_epoch=20,
    class_weight=class_weights,
    callbacks=[
        keras.callbacks.ModelCheckpoint(os.path.join(log_dir, "best_model.h5"), save_best_only=True, monitor='val_accuracy'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3)
    ]
)

# --- SAVE FINAL MODEL & METADATA ---
model.save(os.path.join(log_dir, "plant_disease_model.h5"))
with open(os.path.join(log_dir, 'model_metadata.json'), 'w') as f:
    json.dump({
        'class_names': class_names,
        'image_size': image_size,
        'num_classes': num_classes
    }, f, indent=4)

print("\n✅ Final training complete. Model ready for app.py") 