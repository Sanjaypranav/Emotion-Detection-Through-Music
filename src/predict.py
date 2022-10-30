"""Predict both images and audio"""
import os
import argparse

# run detect.py  in the terminal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='yolov5 or audio', default="yolov5")
    parser.add_argument('--file', type=str, default='test/new-orleans/New Orleans.mp3', help='file name')
    parser.add_argument('--audio_model', type=str, default='ANN', help='Audio Models CNN ANN CNN2 LSTM GRU')
    parser.add_argument('--dir', type=str, default='data/img', help='image dir')
    opt = parser.parse_args()
    print(opt.model)

    if opt.model == 'yolov5':
        os.system(
            f'python src/yolov5/detect.py --weights "src/weights/best.pt" --source {opt.dir} --data "src/config/edm8.yaml" --name results/test')
    if opt.model == 'audio':
        os.system(f'python src/detect.py --file "{opt.file}" --model "{opt.audio_model}"')
