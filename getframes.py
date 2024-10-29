import cv2 as cv
import pandas as pd
import os
from multiprocessing.pool import Pool
from functools import partial
import time

# Take grayscale screenshots of the videos
# 1 screenshot every 3 seconds
# Storing them in a folder under the member's folder, with a folder name of the VID
def getFrames(member : str, vid : str, skipS : int, processNo : int) -> None:
    '''
    Assumes a folder {member}/{vid}/ has already been created
    '''
    rPath = f"{member}/{vid}.mp4"
    cap = cv.VideoCapture(rPath)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    totFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    threads = max(os.cpu_count(), 2)
    framesPerProcess = totFrames//threads

    startFrame = processNo * framesPerProcess
    endFrame = (processNo + 1) * framesPerProcess

    cap.set(cv.CAP_PROP_POS_FRAMES, startFrame)
    ret, frame = cap.read()
    frameNo = startFrame
    while(ret and frameNo < endFrame):
        # Save the image in grayscale
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ssName = str(frameNo).zfill(7)
        cv.imwrite(f"{member}/{vid}/{ssName}.jpg", frame)
        # Skip ahead "skipN" of images
        skipN = fps*skipS # skip 3 seconds of frames
        frameNo += skipN
        cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
        ret, frame = cap.read()
    cap.release()

if __name__ == '__main__':
    # Set max number of threads
    threads = max(os.cpu_count(), 2)

    df = pd.read_excel("tcg_vids.xlsx")
    # List of members used for the prototype
    members = ["Sakura Miko"]
    for mem in members:
        for i in range(0, len(df)):
            if df.iloc[i,0] == mem:
                VIDs = df.iloc[i,2:]
                for vid in VIDs:
                    if pd.isna(vid):
                        break
                    print(f"Working on video {vid}.")
                    vStartTime = time.time()

                    # Create screenshot folder, or assume it already exists
                    try:
                        os.mkdir(f"{mem}/{vid}")
                    except:
                        pass

                    # Setup multiprocessing and use the getFrames function
                    with Pool(threads) as pool:
                        pool.map(partial(getFrames, mem, vid, 3), range(threads))

                    vEndTime = time.time()
                    print(f"Time of {vEndTime - vStartTime}s.")
         

