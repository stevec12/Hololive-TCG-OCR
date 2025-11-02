import youtube_dl as yt
from pathlib import Path
import pandas as pd
import os
import sys

# create functions to download the videos 
# could download them into the "/Captures" folder, then sort into "/Channel_ID" and "Video_ID.mp4"

def vidDownload(excel_path: str, name : str) -> None:
    '''
    Expects a path to an XLSX file with columns: Member Name, Channel ID, Video IDs 1-12
    '''
    df = pd.read_excel(excel_path, header=0, index_col=None)
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

def main() -> int:
    n = len(sys.argv)
    excel = None
    mem = None
    if n < 3:
        excel = input("Enter the path of the Excel file: ")
        mem = input("Enter the member's full English name to download: ")
    else:
        excel = sys.argv[1]
        mem = sys.argv[2]
    vidDownload(excel,mem)
    return 0
    
if __name__ == '__main__':
    main()
    sys.exit()


