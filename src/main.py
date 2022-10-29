from fastapi import FastAPI, File, UploadFile
import io
import os
import sys
import argparse
import pickle as pkl
import tensorflow as tf
import soundfile as sf
import numpy as np
from pydub import AudioSegment

from Audio import Preprocessor
from sklearn.preprocessing import OneHotEncoder

from detect import get_valence_arousal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from rich import print as rprint

app = FastAPI()


# fastapi app to get audio file and return valence and arousal
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model_name = "ANN"
    model = tf.keras.models.load_model(f'models/{model_name}.h5')
    rprint(f"[bold green]Model loaded {model_name}.h5[/bold green]")
    # features = AudioSegment.from_file(io.BytesIO(file.file.read()), format="mp3")
    # print(len(features))
    # preprocess = Preprocessor()
    # features = preprocess.get_features(file.file.read())
    # features = features.reshape(1, -1)
    # probablities = model.predict(features)
    return "to be implemeted "


@app.get("/")
def read_root():
    return {"Hello": "World"}
