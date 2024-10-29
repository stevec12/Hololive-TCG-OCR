import cv2 as cv
import os
from pathlib import Path

testFolder = "test2"
try:
    os.mkdir(testFolder)
except:
    pass

cap = cv.VideoCapture("v1.mp4")

fps = 30
start_frame = fps*5 # number of seconds to initially skip
cap.set(cv.CAP_PROP_FRAME_COUNT, start_frame)

ret, frame = cap.read()
frameNo = start_frame

while(ret):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ssName = str(frameNo).zfill(7)
    cv.imwrite(f"{testFolder}/{ssName}.jpg", frame)
    # Skip ahead "skipN" of images
    skipN = fps*3 # skip 3 seconds of frames
    frameNo += skipN
    cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
    ret, frame = cap.read()

cap.release()
