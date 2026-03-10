import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. SETTINGS
train_dir = 'train'
test_dir = 'test'
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5 # Keep it low (5) for the first test to see if it works

# 2. PREPARE IMAGES
# This resizes images and scales them to numbers between 0 and 1
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading Training Data...")
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary' # 'binary' because we have only 2 classes (Sick vs Healthy)
)

print("Loading Testing Data...")
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 3. BUILD THE MODEL (The Brain)
model = Sequential([
    # First layer finds edges
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    
    # Second layer finds shapes
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third layer finds complex features
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Convert image to list of numbers
    Flatten(),
    
    # The thinking part
    Dense(512, activation='relu'),
    Dropout(0.5), # Prevents memorization
    Dense(1, activation='sigmoid') # Output: 0 or 1 (Sick or Healthy)
])

# 4. COMPILE
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. TRAIN
print("Starting Training... This may take a few minutes.")
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=test_data,
    validation_steps=test_data.samples // BATCH_SIZE
)

# 6. SAVE
model.save('tomato_disease_model.h5')
print("Done! Model saved.")