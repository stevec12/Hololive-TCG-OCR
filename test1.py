import youtube_dl as yt
from pathlib import Path

vidPath = str(Path.cwd()/'v1.mp4')
url = "http://m.youtube.com/watch?v=UF14-3XsuV0"

yt_opts = {'outtmpl': vidPath}
ydl = yt.YoutubeDL(yt_opts)
info_dict = ydl.extract_info(url, download = False)
duration = info_dict['duration']
formats = info_dict.get('formats', None)
for f in formats:
    if f.get('format_note', None) == '480p' and f.get('ext', None) == 'webm':
        u = f.get('url', None)
        print(u)
        ydl.download([u])
