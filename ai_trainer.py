import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. Configuration & Path Setup
DATASET_DIR = os.path.join(os.path.expanduser("~/Desktop"), "Planes_dataset")
IMG_SIZE = (512, 512)
BATCH_SIZE = 10                

print("Loading and preparing your cleaned dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"🚀 Training elite classifier on {num_classes} classes: {class_names}")

# Optimize dataset performance with caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. Enhanced Data Augmentation Pipeline
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),       # Slightly wider angle for varying flight profiles
    layers.RandomZoom(0.08),          
    layers.RandomBrightness(0.12),     
    layers.RandomContrast(0.12),       
], name="data_augmentation")

# 3. Build EfficientNetB2 Model Architecture (Upgraded for sharper feature detection)
print("Building EfficientNetB2 backbone...")
base_model = tf.keras.applications.EfficientNetB2(
    input_shape=(512, 512, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model initially for Phase 1
base_model.trainable = False

inputs = layers.Input(shape=(512, 512, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)          # Slightly higher dropout to prevent memorization
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Compiled with standard SparseCategoricalCrossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Setup path for saving the model to your Desktop as 'airplane_ai2.keras'
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_save_path = os.path.join(desktop_path, "airplane_ai2.keras")

# ModelCheckpoint ensures we always keep the absolute best epoch
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Automatically lower learning rate if validation loss hits a plateau
reduce_lr_callback = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# 4. Phase 1: Training the Head (15 Epochs)
print("\n--- PHASE 1: Training Classification Head (15 Epochs) ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[checkpoint_callback, reduce_lr_callback]
)

# 5. Phase 2: Deep Fine-Tuning (15 Epochs)
print("\n--- PHASE 2: Deep Fine-Tuning Backbone Layers (15 Epochs) ---")
base_model.trainable = True

# Unfreeze the top 60 layers for deeper fine-grained adjustments
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6), # Microscopic learning rate for safe fine-tuning
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[checkpoint_callback, reduce_lr_callback]
)

print(f"\n✨ Elite training pipeline complete! Your upgraded model is saved to your Desktop as:\n{model_save_path}")