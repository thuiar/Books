import numpy as np
from scipy import io


def load_data():
    MAT_FILE = 'data.mat'
    print ('process 0')

    data = io.loadmat(MAT_FILE)

    train_x = data['train_x'].reshape(-1, 1, 64, 64)
    train_y = data['train_y'].reshape(-1)
    print ('process 1')

    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    valid_len = int(len(train_x) / 10)

    train_x = train_x / np.float32(256.0)

    train_x, val_x = train_x[:-valid_len], train_x[-valid_len:]
    train_y, val_y = train_y[:-valid_len], train_y[-valid_len:]

    return (train_x, train_y, val_x, val_y)

train_x, train_y, test_x, test_y = load_data()  
print 'it is done'