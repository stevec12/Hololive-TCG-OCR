# Setup second CNN to filter more clear numbers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from pathlib import Path

# Setup image and dataset attributes
img_height = 360
img_width = 640
batch_size = 8
seed = 1234
SSDir = Path.cwd() / "testss2"

# Load datasets and crop images to relevant part
train_ds = tf.keras.utils.image_dataset_from_directory(SSDir, label_mode='binary', subset='training',
                                                       color_mode='grayscale', batch_size=8, seed=seed, 
                                                       image_size=(img_height,img_width), validation_split=0.2)
train_ds = train_ds.map(lambda img, label: (tf.image.crop_to_bounding_box(img,260,460,30,120), label))
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(SSDir, label_mode='binary', subset='validation', 
                                                      color_mode='grayscale', batch_size=8, seed=seed,
                                                      image_size=(img_height,img_width), validation_split=0.2)
val_ds = val_ds.map(lambda img, label: (tf.image.crop_to_bounding_box(img,260,460,30,120),label))
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16,5,activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(80,activation='relu'),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Softmax()
    ])

model.compile(
    optimizer ='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=7
    )

model.save('secondFilter.keras')