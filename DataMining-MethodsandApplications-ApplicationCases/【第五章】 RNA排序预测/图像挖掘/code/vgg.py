# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:35:55 2016

@author: yuhui
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import numpy as np

#from keras.layers import BatchNormalization

def build_vgg():
    model = Sequential()

    # 64 * 64 * 1
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform', input_shape=(1, 64, 64)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Activation('relu'))
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu',
                            init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 16 * 16 * 8
    model.add(Dropout(0.25))


    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 8 * 8 * 16
    model.add(Dropout(0.25))
    
    
    model.add(Convolution2D(16, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 4 * 4 * 32
    model.add(Dropout(0.25))
    

    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            init='he_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2 * 2 * 64
    model.add(Dropout(0.25))
    

    model.add(Flatten())
    model.add(Dense(128, init='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def load_data():  
    import cPickle as pickle
    with open('x_data.pkl','rb') as f:
        train_x=pickle.load(f)
    with open('y_data.pkl','rb') as f:
        train_y=pickle.load(f)
    train_x = train_x.reshape(-1, 1, 64, 64)
    train_y = train_y.reshape(-1)
    
    index = [i for i in range(len(train_x))]   ###data
    np.random.shuffle(index)
    train_x = train_x[index]
    train_y = train_y[index]
    
    valid_len = int(len(train_x) / 10)
    train_x = train_x / np.float32(255.0)
    train_x, val_x = train_x[:-valid_len], train_x[-valid_len:]
    train_y, val_y = train_y[:-valid_len], train_y[-valid_len:]
    
    return train_x, train_y, val_x, val_y
    
    
    
print(">> Loading Data ...")
train_x, train_y, val_x, val_y = load_data()
print('\\tTherefore, read in', train_x.shape[0], 'samples from the dataset totally.')

print(">> Build Model ...")
model=build_vgg()

print(">> Start Training...")

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
hist = model.fit(train_x, train_y,                 
                batch_size=200,         
                nb_epoch=7,             
                shuffle=True,                 
                verbose=2,                     
                #show_accuracy=True,         
                validation_data=(val_x, val_y),         
                callbacks=[early_stopping])

model.save_weights('practice_weight.hdf5')







