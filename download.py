import youtube_dl as yt
from pathlib import Path
import pandas as pd
import os

# import dataframe from excel file created earlier
df = pd.read_excel("tcg_vids.xlsx", header=0, index_col=None)

# create functions to download the videos 
# could download them into the "/Captures" folder, then sort into "/Channel_ID" and "Video_ID.mp4"

def vidDownload(name : str) -> None:
    '''
    expects a dataframe with columns: member name, channel id, video ids 1-12
    '''
    for i in range(0,len(df)):
        if df.iloc[i,0] == name:
            try:
                os.makedirs(name)
            except: 
                # assume the directory already exists
                pass
            VIDs = df.iloc[i,2:]
            for vid in VIDs: 
                if pd.isna(vid):
                    break
                # Download each video given the vid (video ID)
                rpath = f"{name}/{vid}.mp4"
                vPath = str(Path.cwd()/rpath)
                yt_opts = {'outtmpl': vPath}
                ydl = yt.YoutubeDL(yt_opts)
                url = "http://m.youtube.com/watch?v=" + vid
                try:
                    ydl.download([url])
                except:
                    print(f"Failed to download {vid} for {name}." )

mem = input("Enter the member's full English name to download: ")
vidDownload(mem)


