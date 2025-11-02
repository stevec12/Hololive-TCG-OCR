# Setup second CNN to filter more clear numbers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

def train_secondFilter(trainset_path: str, img_height: int, img_width: int, seed: int, batch_size: int,
                       bd_y: int, bd_x: int, bd_height: int, bd_width: int) -> None:
    # Load datasets and crop images to relevant part
    train_ds = tf.keras.utils.image_dataset_from_directory(trainset_path, label_mode='binary', subset='training',
                                                           color_mode='grayscale', batch_size=8, seed=seed, 
                                                           image_size=(img_height,img_width), validation_split=0.2)
    train_ds = train_ds.map(lambda img, label: (tf.image.crop_to_bounding_box(img,bd_y,bd_x,bd_height,bd_width), label))
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(trainset_path, label_mode='binary', subset='validation', 
                                                          color_mode='grayscale', batch_size=8, seed=seed,
                                                          image_size=(img_height,img_width), validation_split=0.2)
    val_ds = val_ds.map(lambda img, label: (tf.image.crop_to_bounding_box(img,bd_y,bd_x,bd_height,bd_width),label))
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
    
def main(trainset_path : str, img_height=360, img_width=640, bd_y=260, bd_x=460, bd_height=30, bd_width=120) -> int:
    # Setup image and dataset attributes
    batch_size = 8
    seed = 1234
    train_secondFilter(trainset_path, img_height, img_width, seed, batch_size, bd_y, bd_x, bd_height, bd_width)
    return 0
    
    