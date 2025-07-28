from pathlib import Path
import pandas as pd
import cv2
import pytesseract

# vidsDF = pd.read_excel("tcg_vids.xlsx", header=0, index_col=None )
pytesseract.pytesseract.tesseract_cmd = 'D:\\Program Files\\Tesseract\\tesseract.exe'

vid = "7HO6bazzg8U"
vidDir = Path.cwd()/"Usada Pekora"
vidPreFilterPath = vidDir / (vid+"_initial.txt")
vidSSDir = vidDir / vid

filteredSSs = pd.read_csv(vidPreFilterPath, sep=" ", header=None)
filteredSSs.columns = ["ssid","similarity"]
likelySSs = filteredSSs[filteredSSs["similarity"]==1]["ssid"]

count = 0
for ss in likelySSs:
    count +=1
    if count > 100 : break 
    ssPath = vidSSDir / ss
    img_cv = cv2.imread(str(ssPath))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # crop the image to the relevant area
    img_crop = img_cv[260:300, 450:600]

    # remove noise
    img_crop = cv2.GaussianBlur(img_crop, (5,5), 0)
    
    # thresholding
    _, img_crop = cv2.threshold(img_crop, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Tesseract OCR
    lang = '-l eng '
    engines = '--oem 3 --psm 8 '
    blacklist = '-c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyzABCDEFGHOJKLMNOPQRSTUVWXYZ'
    cfg = lang + engines + blacklist
    result = pytesseract.image_to_string(img_crop, config = cfg)
    
    
    print(f"{ss} {result}")