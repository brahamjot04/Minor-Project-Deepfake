import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Enable GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set dataset paths
train_dir = 'Dataset/training/'
val_dir = 'Dataset/val/'
test_dir = 'Dataset/testing/'

# Optimized parameters
img_size = (64, 64)  # Don't go lower than this
batch_size = 16  # Increased batch size for faster training
epochs = 5  # Will stop early if no improvement

# Faster data loading with optimized pipeline
def load_dataset(path, shuffle=True):
    ds = image_dataset_from_directory(
        path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        shuffle=shuffle,
        interpolation='bilinear'  # Faster image resizing
    )
    return ds

train_ds = load_dataset(train_dir, shuffle=True)
val_ds = load_dataset(val_dir, shuffle=False)
test_ds = load_dataset(test_dir, shuffle=False)

# Optimized data pipeline - no caching for large datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Simplified but effective augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Leaner and faster model architecture
def build_model():
    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(*img_size, 3)),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

model = build_model()

# Faster optimizer with higher initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Efficient callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# Training with timing
import time
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    verbose=2  # Cleaner progress output
)

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")

# Quick evaluation
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save model
model.save("face_classifier_optimized.keras")
print("Optimized model saved successfully!")