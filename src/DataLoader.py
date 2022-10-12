import numpy as np
from sklearn.preprocessing import OneHotEncoder
from rich import print as rprint
import pickle as pkl

# load numpy array
X_train = np.load('data/X_train.npy', allow_pickle=True)
X_test = np.load('data/X_test.npy', allow_pickle=True)
y_train = np.load('data/y_train.npy', allow_pickle=True)
y_test = np.load('data/y_test.npy', allow_pickle=True)

# pad sequences
X_train = np.array([x.reshape(40, 1) for x in X_train])
X_test = np.array([x.reshape(40, 1) for x in X_test])

# one hot encode
onehot_encoder = OneHotEncoder()
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = onehot_encoder.transform(y_test.reshape(-1, 1)).toarray()

# save numpy array
np.save('data/X_train.npy', X_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

pkl.dump(onehot_encoder, open('data/onehot_encoder.pkl', 'wb'))
rprint("[bold red]Data preprocessed[/bold red]")