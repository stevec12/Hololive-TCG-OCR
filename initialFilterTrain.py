# Setup a CNN to filter more likely candidates for OCR 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

def main(train_path: str, img_height=360, img_width=640) -> int:
    batch_size = 8
    seed = 1234
    screenshot_url = train_path
    train_ds = tf.keras.utils.image_dataset_from_directory(screenshot_url, label_mode='binary', subset="training",
                                                           color_mode='grayscale', batch_size= batch_size, seed=seed, 
                                                           image_size=(img_height,img_width), validation_split=0.15)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(screenshot_url, label_mode='binary', subset="validation",
                                                           color_mode='grayscale', batch_size= batch_size, seed=seed, 
                                                           image_size=(img_height,img_width), validation_split=0.15)
    
    train_ds = train_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 4, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(320, activation='relu'),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Softmax()
        ])
    
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
        )
    
    model.save('initialFilter.keras')
    return 0