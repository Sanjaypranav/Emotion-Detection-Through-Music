import numpy as np
import pickle as pkl
from model import Model
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
    parser.add_argument('-d', '--data', help='dir containing npy file to train', required=True)
    args = parser.parse_args()
    # load numpy array
    if os.path.exists(args.data):
        X_train = np.load(os.path.join(args.data, 'X_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(args.data, 'X_test.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data, 'y_train.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(args.data, 'y_test.npy'), allow_pickle=True)

        #Load onehot encoder
        onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))

        model = Model(num_labels=2)
        print(model.build_model())
        print(model.summary())
        print(model.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32))
        print(model.save('models/model.h5'))
        print(model.evaluate(X_test, y_test))
        # print(model.plot_model())
    else:
        rprint("[bold red]Invalid data directory[/bold red]")