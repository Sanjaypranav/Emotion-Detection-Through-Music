from fastapi import FastAPI, File, UploadFile
import os
import sys
import argparse
import pickle as pkl
import tensorflow as tf
import soundfile as sf
import numpy as np
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
    print(file.file)
    # return get_valence_arousal(model.predict(np.array([Preprocessor().get_features(file.file)]))[0])

@app.get("/")
def read_root():
    return {"Hello": "World"}