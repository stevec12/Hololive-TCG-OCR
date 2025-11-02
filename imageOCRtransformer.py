from pathlib import Path
import pandas as pd
import torch as pt
import tensorflow as tf
import numpy as np
import os
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

"""
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("seq2seq_model_digital/checkpoint-32")

img_height,img_width = 360, 640
crop = (260,460,30,120) # (bottom, left, height, width)
batch_size = 512 # Can adjust based on how much available memory
pred_parent_dir = Path.cwd() / "Preds"
"""

def batch_predict(mem_name, processor, batch_size,
                  img_height,img_width,bd_y, bd_x, bd_height, bd_width) -> np.ndarray:
    """
    Given a string of the subfolder name (Youtuber's name here), predict the 
    target text for target videos.
    """
    # Import the pretrained models
    # Using the pretrained processor, and the fine-tuned encoder/decoder
    processor = TrOCRProcessor.from_pretrained(processor)
    # Find last checkpoint to use for model
    checkpoint_dir = "seq2seq_model_digital/"
    checkpoints = [c for c in os.listdir(checkpoint_dir) if "checkpoint-" in c]
    checkpoints.sort(key=lambda c: os.path.getmtime(os.path.join(checkpoint_dir, c)))
    model = VisionEncoderDecoderModel.from_pretrained(checkpoints[-1])
    # Setup screenshot paths
    vidsDir = Path.cwd()/mem_name
    vidNames = [f for f in os.listdir(vidsDir) if '.' not in f]
    # Setup prediction directory
    pred_parent_dir = Path.cwd() / "Preds"
    Path.mkdir(pred_parent_dir, exist_ok=True)
    pred_dir = pred_parent_dir / mem_name
    Path.mkdir(pred_dir, exist_ok=True)
    
    for vid in vidNames:
        # Load the filter results and images into a tensor, then filter to only those that pass the 2nd filter
        vidFilter1Path = vidsDir / (vid + "_initial.txt")
        vidFilter2Path = vidsDir / (vid + "_second.txt")
        ssDir = vidsDir / vid
        
        filter1DF = pd.read_csv(vidFilter1Path, sep=" ", header=None,
                                names=["SSid", "Similarity"], index_col=["SSid"])
        filter2DF = pd.read_csv(vidFilter2Path, sep=" ", header=None, 
                                names=["SSid", "Similarity"], index_col=["SSid"])
        
        img_ds = tf.keras.preprocessing.image_dataset_from_directory(ssDir, labels=None, 
                                                                     color_mode='rgb', 
                                                                     batch_size=512,
                                                                     image_size=(img_height,img_width),
                                                                     shuffle=False)
        img_ds = img_ds.map(lambda img: tf.image.crop_to_bounding_box(img,
                                                                      bd_y, bd_x, bd_height, bd_width))
        batch_no = 1
        preds_list = list() # Store predictions to later write to a file
        for batch in img_ds:
            # Filter the batch by the filters to only predict on likely images
            imgFilter = [bool(filter1DF.at[ss, "Similarity"]==1 and filter2DF.at[ss,"Similarity"]==1)
                         for ss in filter1DF.index.to_list()[(batch_no-1)*batch_size:batch_no*batch_size]]
            imgs = batch.numpy()
            imgs = imgs[imgFilter]
            if(len(imgs)>0):
                imgs_data = processor(imgs, return_tensors="pt").pixel_values
                imgs_ids = model.generate(imgs_data)
                gen_text = processor.batch_decode(imgs_ids, skip_special_tokens=True)
                for pred in gen_text: preds_list.append(pred)
            batch_no += 1
            del(imgs)
            del(batch)
        
        # Prepare DataFrame of the predictions to write to file
        ssIDs = filter2DF[filter2DF["Similarity"]==1].index.to_list()
        predDF = pd.DataFrame({"ssID":ssIDs, "Prediction":preds_list},index=None)
        pred_path = pred_dir / (vid + "_initial_preds.txt")
        predDF.to_csv(pred_path, sep=' ', index=False)
        
        del(filter1DF)
        del(filter2DF)
        del(img_ds)
        
def main(name,
         processor="microsoft/trocr-small-printed",
         img_height=360,img_width=640,
         bd_y=260, bd_x=460, bd_height=30, bd_width=120) -> int:
    batch_size=512
    batch_predict(name, processor, batch_size,
                  img_height,img_width,left_bd,upper_bd,right_bd,lower_bd)
    return 0