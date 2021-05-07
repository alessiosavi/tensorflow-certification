#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:25:07 2021

@author: alessiosavi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# %% CONSTANTS
batch_size = 32
img_height = 224
img_width = 224
data_dir = 'image-classification/flower-classification/flower_photos'
validation_split = 0.2

# %% LOAD IMAGES

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split,
    rescale=1./255
)

train_ds = datagen.flow_from_directory(
    directory=data_dir, target_size=(img_height, img_width), subset='training')

val_ds = datagen.flow_from_directory(directory=data_dir, target_size=(
    img_height, img_width), subset='validation')


# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# n_labels =  len(train_ds.class_names)
n_labels = 5

# %%

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(
        1./255, input_shape=(img_height, img_width, 3)),
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
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
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
    layer.trainable = False


inception_model = tf.keras.Sequential([
    pretrained_model,
    # last_pretrained_layer.output,
    layers.Flatten(),
    layers.Dense(n_labels, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
inception_model.compile(optimizer=optimizer,
                        loss=tf.losses.CategoricalCrossentropy(
                            from_logits=True),
                        metrics=["accuracy"])

inception_model.summary()
history = inception_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)
