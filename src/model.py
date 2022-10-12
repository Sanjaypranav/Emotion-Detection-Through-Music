import tensorflow as tf
from tensorflow.keras.models import Sequential as Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from rich import print as rprint
import numpy as np

# Tensorflow to use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


class Model:
    def __init__(self, num_labels: int = 2):
        self.num_labels = num_labels
        self.model = Sequential()

    def build_model(self):
        self.model.add(Dense(256, input_shape=(40,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.num_labels))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())
        return "Model built"

    def train(self, x_train, y_train, x_test, y_test, epochs: int = 100, batch_size: int = 32):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
        return "Model trained"

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        return metrics.classification_report(y_test, y_pred)

    def save(self, file_name: str):
        self.model.save(file_name)
        return "Model saved"

    def load(self, file_name: str):
        self.model = tf.keras.models.load_model(file_name)
        return "Model loaded"

    def predict(self, x):
        return self.model.predict(x)

    def predict_classes(self, x):
        return self.model.predict_classes(x)

    def get_model(self):
        return self.model

    def summary(self):
        rprint("[bold red]Model Summary[/bold red]")
        return self.model.summary()

    def plot_model(self):
        return tf.keras.utils.plot_model(self.model, show_shapes=True)

    def get_config(self):
        return self.model.get_config()

    def get_weights(self):
        return self.model.get_weights()
