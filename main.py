import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Regularizer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("Training_set.csv")
# print(df.head())

print(data.nunique())
X=[]
y = data['label'].values
# print(y.shape)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

dir = 'train'
for x in data['filename']:
    X.append(cv2.imread(os.path.join(dir,x)))
X=np.array(X)

# print(X.shape)

im = X[0]
plt.imshow(im)

im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.imshow(im)

for i in range(X.shape[0]):
    X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)
plt.imshow(X[0])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()

model.add(Rescaling(1 / 255, input_shape = (224, 224, 3)))

model.add(RandomFlip())
model.add(RandomRotation((-1, 1)))
model.add(RandomZoom(height_factor = 0.2, width_factor = 0.2))
model.add(RandomContrast(0.5))

model.add(Conv2D(16, (5, 5), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (5, 5), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2()))
model.add(Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2()))
model.add(Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2()))

model.add(Dropout(0.2))

model.add(Dense(75, activation = 'softmax'))

model.summary()

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

hst = model.fit(X_train, y_train, epochs = 35, batch_size = 100, validation_data = (X_val, y_val))
acc = hst.history['accuracy']
val_acc = hst.history['val_accuracy']
n = len(acc)

plt.ylim(0.0, 1.0)

plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
model.evaluate(X_train, y_train)    
model.evaluate(X_val, y_val)
