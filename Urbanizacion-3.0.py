#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:33:54 2024

@author: cmss-alien
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define paths
img_folder = '/home/cmss-alien/Descargas/urbanización/images'
csv_file = '/home/cmss-alien/Descargas/urbanización/urbanizacion.csv'

# Load and parse the CSV file
metadata = pd.read_csv(csv_file)
metadata['date'] = pd.to_datetime(metadata['date'], format='%d/%m/%Y')  # Specify the correct date format
metadata = metadata.sort_values(by='date')

# Function to load and preprocess images
def load_and_preprocess_images(img_folder, metadata):
    images = []
    for idx, row in metadata.iterrows():
        img_path = os.path.join(img_folder, row['Image_name '])
        img = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure 3 channels
        img = img.resize((256, 256))  # Resize to a consistent size
        img = np.array(img) / 255.0  # Normalize pixel values
        images.append(img)
    return np.array(images)

# Load and preprocess images
images = load_and_preprocess_images(img_folder, metadata)

# Prepare sequences
sequence_length = 5  # Number of images in each sequence
sequences = []
targets = []

for i in range(len(images) - sequence_length):
    sequences.append(images[i:i + sequence_length])
    targets.append(images[i + sequence_length])

sequences = np.array(sequences)
targets = np.array(targets)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# Define the model
def create_model(input_shape):
    model = Sequential()
    
    # Encoder: CNN for feature extraction
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM for sequence modeling
    model.add(LSTM(256, return_sequences=False))
    
    # Decoder: CNN for image generation
    model.add(Dense(128 * 16 * 16, activation='relu'))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    
    return model

input_shape = (sequence_length, 256, 256, 3)
model = create_model(input_shape)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")

# Predict the next image in the sequence
last_sequence = sequences[-1].reshape(1, sequence_length, 256, 256, 3)
predicted_image = model.predict(last_sequence)

# Save the predicted image
predicted_image = (predicted_image * 255).astype(np.uint8)
predicted_image = Image.fromarray(predicted_image.squeeze())
predicted_image.save('predicted_urban_growth.png')

# Function to calculate urban area
def calculate_urban_area(image, threshold=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0  # Convert PIL Image to NumPy array and normalize
    urban_pixels = np.sum(image > threshold)
    total_pixels = image.size
    urban_area_percentage = (urban_pixels / total_pixels) * 100
    return urban_area_percentage

# Calculate urban area for each image
urban_areas = [calculate_urban_area(img) for img in images]

# Calculate urban area change over time
urban_area_change = []
for i in range(1, len(urban_areas)):
    change = urban_areas[i] - urban_areas[i - 1]
    urban_area_change.append(change)

# Calculate predicted urban area
predicted_image_array = np.array(predicted_image) / 255.0  # Convert PIL Image to NumPy array and normalize
predicted_urban_area = calculate_urban_area(predicted_image_array)
predicted_change = predicted_urban_area - urban_areas[-1]

# Print the results
print(f"Urban Area Change Over Time: {urban_area_change}")
print(f"Predicted Urban Area Change: {predicted_change:.2f}%")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()