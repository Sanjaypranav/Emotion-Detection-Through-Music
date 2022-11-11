"""Predict both images and audio"""
import os
import subprocess
import argparse
import numpy as np
import math
import cv2
import moviepy.editor as mp
from rich import print as rprint
import tensorflow as tf
import pickle as pkl
from Audio import Preprocessor
from detect import get_valence_arousal
from numba import cuda
device = cuda.get_current_device()
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
path = 'src/yolov5/runs/detect/exp/labels'


def read_text_file(file_path: str) -> str:
    for file in os.listdir(file_path):
        # print(file)
        with open(f'{file_path}/{file}', "r") as f:
            emotions[int(f.read(1))] += 1
    return f"Done {file_path}"

def happy_or_frail(percentage : list) -> str:
    healthy=["happy","relax","excited"]
    frail=["sad","worry","bored","fear","anger"]
    if classes[percentage.index(max(percentage))] in healthy:
        return "Healthy"
    else:
        return "Frail"

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
    # os.system(f"python src/detect.py -m {opt.audio_model} -f {opt.save_dir}/test.mp3 ")
    # # capture bash output to a text file
    model = tf.keras.models.load_model(f'models/{audio_model}.h5')
    rprint(f"[bold green]Model loaded {audio_model}.h5[/bold green]")
    preprocessor = Preprocessor()
    features = preprocessor.get_features(save_dir + '/' + 'test.mp3')
    features = features.reshape(1, -1)
    onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
    prediction = model.predict(features)
    rprint(f"[bold purple]{onehot_encoder.inverse_transform(prediction)[0][0]}[/bold purple]")
    rprint(f"[bold yellow]{get_valence_arousal(prediction[0]), prediction[0]}[/bold yellow]")
    rprint("[bold green]Prediction complete[/bold green]")
    device.reset()
    # os.system(
    #     "python3 'src/yolov5/detect.py' --weights 'src/weights/best.pt' --source 'results/frames/' --data src/config/edm8.yaml --save-txt")
    read_text_file(path)
    percentage = [i / sum(emotions) for i in emotions]
    rprint(percentage)
    # print(max(percentage))
    print(classes[percentage.index(max(percentage))])
    print(happy_or_frail(percentage))
    rprint(f"[bold green]Done[/bold green]")
    # read_and_count_output_file()
