# https://github.com/Ashutosh-4485/trocr-custom-fine-tune/blob/main/FineTune_TrOCR_on_Custom_data.ipynb

import os
import torch
import evaluate
import numpy as np
import pandas as pd
import glob as glob

from PIL import Image
import zipfile

from dataclasses import dataclass
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
    )

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device('cpu')
# Only using a training set, as it is quite small 
df = pd.read_csv("OCR Pretrains/imgs_labels.txt", sep=" ", header=None, names=["ssID", "Label"], index_col=None)

# Augmentations to the image

(left, upper, right, lower) = (460,260,580,290)
def train_transforms(img):
    # Also, setup an image cropping specific to this task
    (left, upper, right, lower) = (460,260,580,290)
    img = img.crop((left, upper, right, lower))
    img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    return img

# Training and Dataset Config
@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 10
    EPOCHS: int = 12
    LEARNING_RATE: float = 0.00005

@dataclass(frozen=True)
class DatasetConfig:
    DATA_ROOT: str = 'D:/Side Projects D/repo/TCG Shop Stream OCR/Solution1/OCR Pretrains/imgs/'
    
@dataclass(frozen=True)
class ModelConfig:
    MODEL_NAME: str = "microsoft/trocr-small-printed"

class CustomOCRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=16):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        file_name = self.df['ssID'].iloc[idx]
        text = self.df['Label'].iloc[idx]
        img = Image.open(self.root_dir+str(file_name)).convert('RGB')
        img = train_transforms(img)
        pixel_values = self.processor(img, return_tensors='pt').pixel_values
        labels = self.processor.tokenizer(text,padding='max_length',max_length=self.max_target_length).input_ids
        # Using -100 as the padding token
        labels = [label if label !=self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels":torch.tensor(labels)}
        return encoding
 
processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)
train_dataset = CustomOCRDataset(
    root_dir=os.path.join(DatasetConfig.DATA_ROOT),
    df=df,
    processor=processor
    )       

# Dataset and Image Previews     
encoding = train_dataset[0]
for k,v in encoding.items():
    print(k, v.shape)

image = Image.open(train_dataset.root_dir+str(df['ssID'][0])).convert("RGB")
image = train_transforms(image)
print(image.size)
# image.show()

labels = encoding['labels']
labels[labels==-100]=processor.tokenizer.pad_token_id
label_str=processor.decode(labels, skip_special_tokens=True)
print(label_str)

# Freeze Decoder, Train Only Encoder
model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
model.to(device)

# Freeze decoder parameters, language model head, 
for param in model.decoder.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters (encoder only).")

# Model Config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
# Optimizer Config
optimizer = optim.AdamW(
    model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=0.0005
)

# Evaluation Metric
cer_metric = evaluate.load('cer')

def compute_cer(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}

# Training and Validation Loops
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
    fp16=True,
    output_dir='seq2seq_model_digital/',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    num_train_epochs=TrainingConfig.EPOCHS
    )

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_cer,
    train_dataset=train_dataset,
    data_collator=default_data_collator
    )

# Train
res = trainer.train()
res
res.global_step

# Save Model checkpoint
def zip_folder(folder_path, zip_path):
  with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(folder_path):
      for file in files:
        zipf.write(os.path.join(root, file),
                   os.path.relpath(os.path.join(root, file),
                                   os.path.join(folder_path, '..')))
            
zip_folder('seq2seq_model_digital/checkpoint-'+str(res.global_step), 
           'seq2seq_model_digital/checkpoint-'+str(res.global_step)+'.zip')
