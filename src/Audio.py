import io

import librosa
import librosa.display
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
from rich import print as rprint


class Preprocessor:
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 40) -> object:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def get_features(self, file_name: str):
        # if file is in IO bytes
        if type(file_name) == bytes:
            rprint("[bold red]File is in bytes[/bold red]")
            audio, sample_rate = librosa.load(io.BytesIO(file_name), res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        else:
            try:
                audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
                mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
                mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            except Exception as e:
                rprint(f"[bold red]Error encountered while parsing file [/bold red]")
                return None
        return mfccs_scaled_features

    def plot_features(self, file_name: str):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs_features, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
