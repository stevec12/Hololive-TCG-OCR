import cv2 as cv
import pandas as pd
import os

df = pd.read_excel("tcg_vids.xlsx")
# List of members used for the prototype
test_members = ["Kazama Iroha"]

# Take grayscale screenshots of the videos
# 1 screenshot every 3 seconds
# Storing them in a folder under the member's folder, with a folder name of the VID

for test_mem in test_members:
    for i in range(0, len(df)):
        if df.iloc[i,0] == test_mem:
            VIDs = df.iloc[i,2:]
            for vid in VIDs:
                if pd.isna(vid):
                    break
                print(f"Working on video {vid}.")
                # Load video and skip start
                rpath = f"{test_mem}/{vid}.mp4"
                cap = cv.VideoCapture(rpath)
                fps = 30
                start_frame = fps*600 # number of seconds to initially skip
                cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
                
                # Create screenshot folder, or assume it already exists
                try:
                    os.mkdir(f"{test_mem}/{vid}")
                except:
                    pass

                # Take grayscale screenshots
                ret, frame = cap.read()
                frameNo = start_frame
                while(ret):
                    # Save the image in grayscale
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    ssName = str(frameNo).zfill(7)
                    cv.imwrite(f"{test_mem}/{vid}/{ssName}.jpg", frame)
                    # Skip ahead "skipN" of images
                    skipN = fps*3 # skip 3 seconds of frames
                    frameNo += skipN
                    cap.set(cv.CAP_PROP_POS_FRAMES, frameNo)
                    ret, frame = cap.read()

                cap.release()
         

