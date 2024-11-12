# testing initialFilter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import numpy as np

path = "initialFilter.keras"
model = tf.keras.models.load_model(path)

img_height = 360
img_width = 480

member = 'Kazama Iroha'
vids = [f for f in os.listdir(member) if '.' not in f]


def pred(model, vid):
    vdir = f"{member}/{vid}"
    images = tf.keras.preprocessing.image_dataset_from_directory(vdir, labels=None, 
                                                                      color_mode='grayscale', image_size=(img_height, img_width),
                                                                      shuffle=False)
    
    preds = model.predict(images)
    threshold = 0.25
    preds = preds[:,-1] > threshold # Percent to predict as a ss that should be labelled 
    preds = np.where(preds, 1, 0)
    preds = preds.astype(dtype='S16')

    fileNames = os.listdir(vdir)

    predsWithNames = np.stack((fileNames, preds), axis=1)
    save_path = f"{member}/{vid}_initial.txt"

    np.savetxt(save_path, predsWithNames, fmt='%s')

for vid in vids:
    pred(model, vid)