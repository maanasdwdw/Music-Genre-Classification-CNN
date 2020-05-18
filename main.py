"""
Main code with the convolutional model.

Commented out code is for loading whole tracks. We are instead loading the sample arrays generated by the preprocessing file.
"""
import numpy as np
from scipy.io import wavfile
path = '/music/'
hop_length = 512
import pandas as pd
from sklearn.utils.multiclass import unique_labels

df = pd.read_csv('/music/genres.csv')
k = unique_labels(df['label'])
k = k.tolist()
import librosa
from tqdm import tqdm

#mfccs_all = []
#spectral_center_all = []
#chroma_stft_all = []
#spectral_contrast_all = []
#
#label=[]
#
#def load_into_array(path):
#    for file in tqdm(df['fname']):
#        y, sr = librosa.load(path + file, duration=28)
#
#        mfcc = load_mfcc(y,sr)
#        spectral_center = load_spect(y,sr)
#        chroma_stft = load_chroma_stft(y,sr)
#        spectral_contrast = load_spectral_contrast(y,sr)
#
#        mfccs_all.append(mfcc)
#        spectral_center_all.append(spectral_center)
#        chroma_stft_all.append(chroma_stft)
#        spectral_contrast_all.append(spectral_contrast)
#        s=''
#        label.append(k.index(s.join(df.loc[df['fname']==file].label.values)))

label_sam=[]
mfccs_all_sam = []
spectral_center_all_sam = []
chroma_stft_all_sam = []
spectral_contrast_all_sam  = []

mfccs_all_sam = np.load('/data/mfccs_all_sam.npy')
chroma_stft_all_sam = np.load('/data/chroma_stft_all_sam.npy')
spectral_contrast_all_sam = np.load('/data/spectral_contrast_all_sam.npy')
spectral_center_all_sam = np.load('/data/spect_imgstral_center_all_sam.npy')
label_sam = np.load('/data/label_sam.npy')

# Concatenating the features.
import cv2
concat_imgs = []
for i in tqdm(range(len(mfccs_all_sam))):
    concat_imgs.append(np.vstack((chroma_stft_all_sam[i],mfccs_all_sam[i],spectral_contrast_all_sam[i],spectral_center_all_sam[i])))

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Preparing and splitting data for the model.
label_sam = np.array(label_sam)
label_sam = to_categorical(label_sam)

def img_to_model(array,label):
    array = np.array(array)
    array = array.reshape(array.shape[0], array.shape[1], array.shape[2], 1)
    X_train, X_test, y_train, y_test = train_test_split(array, label, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, X_val, y_val
    
X_train, X_test, y_train, y_test, X_val, y_val = img_to_model(concat_imgs,label_sam)
input_shape = X_train[0].shape

import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, Dropout, Activation
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization


from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

# Convolutional model.
def conv_model(model_input):
    layer = model_input
    conv_1 = Conv2D(filters = 96, kernel_size = (11,11), strides=(4,4),padding= 'same', activation='relu', name='conv_1')(layer)
    pool_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv_1)
    bn1 = BatchNormalization()(pool_1)
    
    conv_2 = Conv2D(filters = 256, kernel_size = (11,11), strides=(1,1),padding= 'same', activation='relu', name='conv_2')(bn1)
    pool_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv_2)
    bn2 = BatchNormalization()(pool_2)
    
    conv_3 = Conv2D(filters = 384, kernel_size = (3,3), strides=(1,1),padding= 'same', activation='relu', name='conv_3')(bn2)
    bn3 = BatchNormalization()(conv_3)
    
    conv_4 = Conv2D(filters = 384, kernel_size = (3,3), strides=(1,1),padding= 'same', activation='relu', name='conv_4')(bn3)
    bn4 = BatchNormalization()(conv_4)

    conv_5 = Conv2D(filters = 256, kernel_size = (3,3), strides=(1,1),padding= 'same', activation='relu', name='conv_5')(bn4)
    pool_3 = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(conv_5)
    bn5 = BatchNormalization()(pool_3)
    
    flatten1 = Flatten()(bn5)
    
    dense1 = Dense(4096,input_shape=(224*224*3,),activation='relu')(flatten1)
    do1 = Dropout(0.4)(dense1)
    bn6 = BatchNormalization()(do1)
    
    dense2 = Dense(4096,activation='relu')(bn6)
    do2 = Dropout(0.4)(dense2)
    bn7 = BatchNormalization()(do2)
    
    dense3 = Dense(1000,activation='relu')(bn7)
    do3 = Dropout(0.4)(dense3)
    bn8 = BatchNormalization()(do3)
    
    output = Dense(10,activation='softmax')(bn8)
    
    model_output = output
    model = Model(model_input, model_output)
    
    model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])
    print(model.summary())
    return model
    
def train_model(x_train, y_train, x_val, y_val):
    
    n_frequency = 5
    n_frames = 33
    
    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')
    
    model = conv_model(model_input)
    checkpoint_callback = ModelCheckpoint('weights.best.h5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
    
    reducelr_callback = ReduceLROnPlateau(
                monitor='val_acc', factor=0.5, patience=10, min_delta=0.01,
                verbose=1
            )
    callbacks_list = [checkpoint_callback, reducelr_callback]

    # Fit the model and get history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=7000, epochs=150,
                        validation_data=(x_val, y_val), verbose=1, callbacks=callbacks_list)

    return model, history

model, history  = train_model(X_train, y_train, X_val, y_val)
model.save("/saved_model/model_150epoch.h5")