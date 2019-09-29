import numpy as np
from keras.utils import to_categorical

def create_labels(train_positives, train_negatives=None, flag=False):
    ''' This function creates labels for model training '''    
    if flag == False :
        # only positive data in trainings
        labels = np.zeros(train_positives.shape[0])
        labels[:] = 1    
    else:
        # negatives & positives data in training
        labels = np.zeros(train_positives.shape[0] + train_negatives.shape[0])
        labels[:train_positives.shape[0]] = 1
    return np.expand_dims(labels, axis=1)

def reshape_X(data, nrows, ncols):
    data_t = np.zeros((data.shape[0], nrows, ncols))
    data_cols = data[0].shape[0]-1
    ctr = 0
    for i, j in zip(range(0, data_cols//2, 2), range(data_cols//2, data_cols, 2)):
        data_t[:, ctr, :] = np.hstack([data[:, i:i+2], data[:, j:j+2]])
        ctr += 1
    return data, data_t

def reshape_y(y, nrows):
    y = to_categorical(y)
    print("\ny shape : ", y.shape)
    y_ = np.zeros((nrows, y.shape[0], y.shape[1]))
    for i in range(nrows):
        y_[i, :, :] = y
    return y_
    
def split_train_validation(x, y, val_split=0.1):
    m = x.shape[0]
    val_size = int(0.1 * m)
    return x[:-val_size], y[:, :-val_size, :], x[-val_size:], y[:, -val_size:, :]

