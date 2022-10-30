import os
import argparse
# run detect.py  in the terminal

# get user to predict yolov5 or audio
# if audio, get user to input file name
# if yolov5, get user to input image name

# if audio, run predict.py
# if yolov5, run detect.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5', help='yolov5 or audio')
    parser.add_argument('--file', type=str, default='test/new-orleans', help='file name')
    parser.add_argument('--dir', type=str, default='data/img', help='image dir')
    opt = parser.parse_args()
    print(opt)

    if opt.model == 'yolov5':
        os.system(f'python src/yolov5/detect.py --weights "src/weights/best.pt" --source {opt.dir} --data "src/config/edm8.yaml" --name results/test')
    elif opt.model == 'audio':
        os.system(f'python predict.py --file {opt.file}')