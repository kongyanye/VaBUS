import os

import pytube
from pytube.helpers import safe_filename

download_list = [
    'https://www.youtube.com/watch?v=MNn9qKG2UFI',
    'https://www.youtube.com/watch?v=QuUxHIVUoaY',
    'https://www.youtube.com/watch?v=wqctLW0Hb_0',
    'https://www.youtube.com/watch?v=5_XSYlAfJZM',
    'https://www.youtube.com/watch?v=jjlBnrzSGjc',
    'https://www.youtube.com/watch?v=1EiC9bvVGnk',
    'https://www.youtube.com/watch?v=WxgtahHmhiw'
]

if __name__ == '__main__':
    for url in download_list:
        yt = pytube.YouTube(url)
        if os.path.exists(f'{safe_filename(yt.title)}.mp4'):
            print(f'skipping "{yt.title}"')
            continue
        if yt.length == 0:
            print(f'failed to download live stream video {url}, \
                use the following command to download and transform:')
            print('ffmpeg -i $(youtube-dl -f {format_id} \
            -g <youtube URL>) -c copy -t 00:20:00 {VDIEONAME}.ts')
            print('ffmpeg -i {VDIEONAME}.ts -acodec copy -vcodec \
            copy -f mp4 {VDIEONAME}.mp4')
        else:
            print(f'downloading "{yt.title}" ({url})...')
            yt.streams.filter(file_extension='mp4').order_by(
                'resolution').desc().first().download()
