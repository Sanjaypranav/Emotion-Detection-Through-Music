import os
import warnings
warnings.filterwarnings('ignore')
import argparse
from Audio import Preprocessor
from rich import print as rprint
import pickle as pkl
import tensorflow as tf
# load audio file



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
        features = features.reshape(1, -1)
        onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
        prediction = onehot_encoder.inverse_transform(model.predict(features))
        print(prediction[0][0])
        rprint("[bold green]Prediction complete[/bold green]")
    else:
        rprint("[bold red]Invalid file[/bold red]")
