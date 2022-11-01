"""Predict both images and audio"""
import os
import subprocess
import argparse
import numpy as np
import math
import cv2
import moviepy.editor as mp
from rich import print as rprint

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


# run detect.py  in the terminal
def audio_extractor(file_path: str, save_dir: str = "test/tiktok/") -> str:
    video_clip = mp.VideoFileClip(file_path)  # change path
    video_clip.audio.write_audiofile(save_dir + "test.mp3")
    return "Audio Extracted"


def video_breaker(video_path: str, save_dir: str = "results/frames/") -> str:
    # using cv2
    capture = cv2.VideoCapture(video_path)
    frameRate = 0  # frame rate
    while (True):
        success, frame = capture.read()
        if success:
            cv2.imwrite(f'{save_dir}/{frameRate}.jpg', frame)  # change destination

        else:
            break
        frameRate = frameRate + 1
    capture.release()
    return "Frames Saved"


classes = {
    0: "anger",
    1: "bored",
    2: "excited",
    3: "fear",
    4: "happy",
    5: "relax",
    6: "sad",
    7: "worry"
}

global emotions, path
emotions = [0, 0, 0, 0, 0, 0, 0, 0]
path = 'src/yolov5/runs/detect/exp17/labels'


def read_text_file(file_path: str) -> str:
    for file in os.listdir(file_path):
        # print(file)
        with open(f'{file_path}/{file}', "r") as f:
            emotions[int(f.read(1))] += 1
    return f"Done {file_path}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='path to video file', required=True)
    parser.add_argument('--audio_model', type=str, default='ANN', help='Audio Models CNN ANN CNN2 LSTM GRU')
    parser.add_argument('--save_dir', type=str, default="test/tiktok/", help="Save dir to save extracted audio")
    opt = parser.parse_args()
    video_path = opt.video
    audio_model = opt.audio_model
    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        print("Creating directory")
        os.makedirs(save_dir)
    rprint(f"[bold green]Extracting audio from {video_path}[/bold green]")
    print(audio_extractor(video_path, save_dir))
    print(video_breaker(video_path))
    os.system(f"python src/detect.py -m {opt.audio_model} -f {opt.save_dir}/test.mp3 ")
    # # capture bash output to a text file
    os.system(
        "python3 'src/yolov5/detect.py' --weights 'src/weights/best.pt' --source 'results/frames/' --data src/config/edm8.yaml --save-txt")
    read_text_file(path)
    percentage = [i / sum(emotions) for i in emotions]
    rprint(percentage)
    # print(max(percentage))
    print(classes[percentage.index(max(percentage))])
    # read_and_count_output_file()
