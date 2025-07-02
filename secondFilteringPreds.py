import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd

model_path = 'secondFilter.keras'
model = tf.keras.models.load_model(model_path)

img_height = 360
img_width = 640

member = 'Usada Pekora'
vids = [f for f in os.listdir(member) if '.' not in f]

def secondFilter(model, vid):
    vdir = f"{member}/{vid}"
    initFilterDir = f"{member}/{vid}_initial.txt"
    initFilter = pd.read_csv(initFilterDir, sep=" ", header=None, names=["vID","Likely"])
    
    # Import the images from the directory as a TF dataset
    images = tf.keras.preprocessing.image_dataset_from_directory(vdir, labels=None, 
                                                                 color_mode='grayscale', 
                                                                 batch_size=None,
                                                                 image_size=(img_height,img_width),
                                                                 shuffle=False)

    # Convert the dataset into a NP array
    images_np = np.array(list(images.as_numpy_iterator()))
    
    # Form a filter for the dataset to only test imgs that passed the 1st filter
    firstFilterBools = [False if result == 0 else True for result in initFilter["Likely"]]
    # Store the filenames of the relevant images
    filteredImgNames = initFilter[initFilter["Likely"]==1]["vID"]
    
    images_np = images_np[firstFilterBools]
    images = tf.data.Dataset.from_tensor_slices(images_np)
    images = images.batch(32)
    images = images.map(lambda img: tf.image.crop_to_bounding_box(img,260,460,30,120))
    
    # Make predictions on the filtered images
    preds=model.predict(images)
    
    threshold = 0.75 # A high threshold to ensure only easily readable images recieve OCR later
    preds = preds[:,-1] > threshold
    preds = np.where(preds, 1, 0)
    preds = preds.astype(dtype='S16')
    
    # Create dataframe with the filtered video names and the second filter results
    filteredImgNames = initFilter[initFilter["Likely"]==1]["vID"].to_numpy(dtype='<U16')
    
    predsWithNames = np.stack((filteredImgNames, preds), axis=1)
    save_path = f"{member}/{vid}_second.txt"
    
    np.savetxt(save_path, predsWithNames, fmt='%s')

for vid in vids:
    secondFilter(model, vid)



