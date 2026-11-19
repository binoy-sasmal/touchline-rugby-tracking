"""Explicit downloads only; ordinary inference/test commands never access the network."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
MODEL_SHA='c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
VIDEO_SHA='7826343d64a2b945e62f9a966951bd786838c4401857237ecd1e7dbce135826f'
REV='48b10fa505aeb51fbd127fc60a5c660040e823b7'


def check(path,expected):
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError(f'{path.name}: checksum changed; inspect source and do not reuse frozen labels without verifying frame correspondence')
    print(path.name,actual)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true');args=parser.parse_args()
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    model,video=assets/'yolox.onnx',assets/'rugby.mp4'
    if not model.exists() and args.download:
        url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{REV}/models/object_detection_yolox/object_detection_yolox_2022nov.onnx'
        urllib.request.urlretrieve(url,model)
    if not video.exists() and args.download:
        subprocess.run([sys.executable,'-m','yt_dlp','--no-playlist','--retries','0','--js-runtimes','node','-f','134','-o',str(video),'https://www.youtube.com/watch?v=CXWyq9hcZTg'],check=True)
    for path,expected in [(model,MODEL_SHA),(video,VIDEO_SHA)]:
        if not path.is_file():raise FileNotFoundError(f'{path}: run prepare_assets.py --download explicitly, or supply the recorded asset')
        check(path,expected)


if __name__=='__main__':main()
