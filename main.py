import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("Training_set.csv")
X = []
y = data['label'].values

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Load images
dir = 'train'
for x in data['filename']:
    X.append(cv2.imread(os.path.join(dir, x)))
X = np.array(X)

# Preprocess images
for i in range(X.shape[0]):
    X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Rescaling(1. / 255, input_shape=(224, 224, 3)))
model.add(RandomFlip())
model.add(RandomRotation((-1, 1)))
model.add(RandomZoom(height_factor=0.2, width_factor=0.2))
model.add(RandomContrast(0.5))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2()))
model.add(Dropout(0.2))
model.add(Dense(75, activation='softmax'))

# Print model summary
# model.summary()

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint callback
checkpoint_path = "training_checkpoint/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      save_weights_only=True,
                                      save_best_only=True,
                                      verbose=1,
                                      monitor='val_accuracy',
                                      mode='max')

if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print("Checkpoint weights loaded successfully!")

# Define EarlyStopping
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with checkpoint callback
history = model.fit(X_train, y_train,
                    epochs=2,  # Example: train for 10 epochs
                    batch_size=100,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint_callback, early_stopping_callback])

# Evaluate model on training and validation sets
model.evaluate(X_train, y_train)
model.evaluate(X_val, y_val)

# Load testing data
data_test = pd.read_csv("Testing_set.csv")
X_test = []
dir = "test"

# Load test images
for x in data_test['filename']:  # Use data_test here
    X_test.append(cv2.imread(os.path.join(dir, x)))
X_test = np.array(X_test)

# Preprocess test images
for i in range(X_test.shape[0]):
    X_test[i] = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2RGB)

# Make predictions
y_pred = model.predict(X_test)
predictions = np.argmax(y_pred, axis=1)
predictions = encoder.inverse_transform(predictions)

print(predictions)
