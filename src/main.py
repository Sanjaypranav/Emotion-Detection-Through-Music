from fastapi import FastAPI, File, UploadFile
import os
import pickle as pkl
import tensorflow as tf
from numba import cuda

device = cuda.get_current_device()
from Audio import Preprocessor

from detect import get_valence_arousal
from predict import audio_extractor, video_breaker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from rich import print as rprint

app = FastAPI()

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


# fastapi app to get audio file and return valence and arousal
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"data/scraped-videos/{file.filename}"
    print(file_path)
    # Extract audio from video
    audio_extractor(file_path)
    # Break video into frames
    video_breaker(file_path)
    # Preprocess audio
    preprocessor = Preprocessor()
    preprocessor = Preprocessor()
    features = preprocessor.get_features('test/tiktok/test.mp3')
    features = features.reshape(1, -1)
    # Load model
    model = tf.keras.models.load_model('models/ANN.h5')
    # Predict valence and arousal
    onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
    prediction = model.predict(features)
    rprint(f"[bold purple]{onehot_encoder.inverse_transform(prediction)[0][0]}[/bold purple]")
    rprint(f"[bold yellow]{get_valence_arousal(prediction[0]), prediction[0]}[/bold yellow]")
    rprint("[bold green]Prediction complete[/bold green]")
    device.reset()
    os.system(
        "python3 'src/yolov5/detect.py' --weights 'src/weights/best.pt' --source 'results/frames/' --data src/config/edm8.yaml --save-txt")
    read_text_file(path)
    print(emotions)
    percentage = [i / sum(emotions) for i in emotions]
    rprint(percentage)
    return f"{classes[percentage.index(max(percentage))]} % {max(percentage) * 100}"


@app.get("/")
def read_root():
    return {"Hello": "World"}
