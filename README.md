# App Description
Performs OCR on YouTube videos, extracting relevant text for different purposes. Here, the targetted text is TCG Shop Simulator card values in opened packs. 
Run Order:
1. download.py
2. getframes.py
3. initialFilteringTrain.py (if training a new model)
4. initialFilteringPreds.py
5. imageOCR.py
# TDL 
* Crop the targeted images to the relevant area where the text can occur.
* Load a pretrained OCR model
* Train the model on some human-extracted samples
* Classify the data
* Format the data appropriately for data analysis in Power BI.
