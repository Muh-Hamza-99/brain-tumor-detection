import cv2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

features = []
labels = []

input_dimensions = (64, 64)
images_directory = "data/datasets/"

no_tumor_directory = f"{images_directory}no/"
yes_tumor_directory = f"{images_directory}yes/"

no_tumor_images = os.listdir(no_tumor_directory)
yes_tumor_images = os.listdir(yes_tumor_directory)

for i, image_name in enumerate(no_tumor_images):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(f"{no_tumor_directory}{image_name}")
        image = Image.fromarray(image, "RGB")
        image = image.resize(input_dimensions)
        features.append(np.array(image))
        labels.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split(".")[1] == "jpg":
        image = cv2.imread(f"{yes_tumor_directory}{image_name}")
        image = Image.fromarray(image, "RGB")
        image = image.resize(input_dimensions)
        features.append(np.array(image))
        labels.append(1)

features = np.array(features)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(input_dimensions[0], input_dimensions[1], 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=64))
model.add(Activation("relu"))

model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

model.save("braintumor10epochs.h5")