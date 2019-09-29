import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector, Concatenate
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.optimizers import Adam

def initialize_model(n_in, n_a, n_out, reshape_row, reshape_col):
    
    reshapor = Reshape((reshape_row, reshape_col))
    
    print("reshapor : ",reshapor)
    
    LSTM_cell = LSTM(n_a, return_state = True)
    
    densor = Dense(n_out, activation='softmax')
    
    return reshapor, LSTM_cell, densor

def create_model(Tx, n_in, n_a, n_out, reshape_row, reshape_col):

    X = Input(shape=(Tx, n_in))
    
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    
    a = a0
    c = c0
    
    reshapor, LSTM_cell, densor = initialize_model(n_in, n_a, n_out, reshape_row, reshape_col)
    
    outputs = []

    for t in range(Tx):
         
        x = Lambda(lambda x: X[:, t, :])(X)
        x = reshapor(x)

        a, _, c = LSTM_cell(x, initial_state=[a, c])

        out = densor(a)

        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model
