import numpy as np
import pickle as pkl
from model import ANN, CNN, CNN2, LSTM, GRU
import argparse
from rich import print as rprint
import os

# # load numpy array
# X_train = np.load('data/X_train.npy', allow_pickle=True)
# X_test = np.load('data/X_test.npy', allow_pickle=True)
# y_train = np.load('data/y_train.npy', allow_pickle=True)
# y_test = np.load('data/y_test.npy', allow_pickle=True)
#
# #Load onehot encoder
# onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
#
# model = Model(num_labels=2)
# print(model.build_model())
# print(model.summary())
# print(model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32))
# print(model.save('models/model.h5'))
# print(model.evaluate(X_test, y_test))
# print(model.plot_model())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-d', '--data', help='dir containing npy file to train', required=False, default='data')
    parser.add_argument('-m', '--model_name', help='model to use [ANN, CNN, LSTM, GRU, CNN2]', required=False, default='ANN')
    args = parser.parse_args()
    model_name = args.model_name
    # load numpy array
    if os.path.exists(args.data):
        X_train = np.load(os.path.join(args.data, 'X_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(args.data, 'X_test.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data, 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(args.data, 'y_test.npy'), allow_pickle=True)

        #Load onehot encoder
        onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
        if model_name == 'ANN':
            model = ANN(num_labels=2)
        elif model_name == 'CNN':
            model = CNN(num_labels=2)
        elif model_name == 'CNN2':
            model = CNN2(num_labels=2)
        elif model_name == 'LSTM':
            model = LSTM(num_labels=2)
        elif model_name == 'GRU':
            model = GRU(num_labels=2)
        else:
            rprint("[bold red]Invalid model[/bold red]")
            exit(1)
        print(model)
        print(model.build_model())
        print(model.summary())
        print(model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32))
        print(model.save(f'models/{model_name}.h5'))
        print(model.evaluate(X_test, y_test))
        # print(model.plot_model())
    else:
        rprint("[bold red]Invalid data directory[/bold red]")