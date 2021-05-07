#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:25:07 2021

@author: alessiosavi
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tqdm import tqdm
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# %% CONSTANTS
batch_size = 32
img_height = 75
img_width = 75
validation_split = 0.2

# %%

(ds_train, ds_test), ds_info = tfds.load('mnist',
                                         split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True,
                                         with_info=True)
# %%
x_train, y_train = [], []
x_test, y_test = [], []

for img, label in tqdm(ds_train):
    x_train.append(img.numpy())
    y_train.append(label.numpy())

for img, label in tqdm(ds_test):
    x_test.append(img.numpy())
    y_test.append(label.numpy())


x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

assert len(np.unique(y_test)) == len(np.unique(y_train))

n_labels = len(np.unique(y_test))
# %% LOAD IMAGES

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255)

train_ds = datagen.flow(x_train, y_train)
val_ds = datagen.flow(x_test, y_test)

# Reshape the data to be like an RGB
train_ds_rgb = datagen.flow(np.repeat(x_train, 3, 3), y_train)
val_ds_rgb = datagen.flow(np.repeat(x_test, 3, 3), y_test)


# %%

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(
        img_height, img_width, input_shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_labels, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy'])

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)
# %%
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

pretrained_model = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", input_shape=(img_height, img_width, 3))
# pretrained_model.summary()
for layer in pretrained_model.layers:
    layer.trainable = True


inception_model = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(
        img_height, img_width, input_shape=(28, 28, 3)),
    pretrained_model,
    layers.Flatten(),
    layers.Dense(n_labels, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
inception_model.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=["accuracy"])

inception_model.summary()
history = inception_model.fit(
    train_ds_rgb,
    validation_data=val_ds_rgb,
    epochs=3
)
