import numpy as np
#np.random.seed(12)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.set_random_seed(11)
from keras.layers import Dense, Input, Conv2D, BatchNormalization, Flatten, Concatenate, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras import regularizers
import warnings 
warnings.filterwarnings('ignore')

def mlp(input_shape=None):
    input = Input(shape=input_shape)
    x = Dense(2048, activation='sigmoid')(input)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=0.0001), loss='mae', metrics=['mae'])
    return model

