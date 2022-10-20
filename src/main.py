import os
import warnings
warnings.filterwarnings('ignore')
import argparse
from Audio import Preprocessor
from rich import print as rprint
import pickle as pkl
import tensorflow as tf
import soundfile as sf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load audio file

def Arousal(probablities):
    if probablities[0] > probablities[1]:
        return 'high arousal'
    else:
        return 'low arousal'

def trim_audio_to_10_seconds(audio):
    if len(audio) > 160000:
        return audio[:160000]
    else:
        return audio
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-f', '--file', help='file to predict', required=True)
    parser.add_argument('-m', '--model_name', help='model to use [ANN, CNN, LSTM, GRU, CNN2]', required=False, default='ANN')
    args = parser.parse_args()
    model_name = args.model_name
    # load numpy array
    if os.path.exists(args.file) and os.path.exists('models'+os.sep+model_name+'.h5'):
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        rprint(f"[bold green]Model loaded {model_name}.h5[/bold green]")
        preprocessor = Preprocessor()
        features = preprocessor.get_features(args.file)
        print(len(features))
        # features = trim_audio_to_10_seconds(features)
        features = features.reshape(1, -1)
        onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
        prediction = model.predict(features)
        rprint(f"[bold purple]{onehot_encoder.inverse_transform(prediction)[0][0]}[/bold purple]")
        rprint(f"[bold purple]{Arousal(prediction[0])}[/bold purple]")
        rprint("[bold green]Prediction complete[/bold green]")
    else:
        rprint("[bold red]Invalid file[/bold red]")
