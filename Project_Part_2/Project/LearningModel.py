import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, Dense, BatchNormalization
import keras.backend as K

def PSNR(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)



# input shape is of the form (patch_size, patch_size, channels)
def create_base_SRCNN_model(input_shape, f1, f2, f3, n1, n2):

    SRCNN = Sequential()
    
    SRCNN.add(Conv2D(activation="relu", input_shape=input_shape, kernel_size=f1, filters=n1,
                     padding="same",kernel_initializer='glorot_uniform', use_bias = True))
    SRCNN.add(BatchNormalization())
    SRCNN.add(Conv2D(activation="relu", kernel_size=f2, filters=n2, padding="same",kernel_initializer='glorot_uniform', use_bias = True))
    SRCNN.add(Conv2D(activation="relu",kernel_size=f3, filters=3, padding="same",kernel_initializer='glorot_uniform', use_bias = True))
    SRCNN.add(BatchNormalization())
    
    SRCNN.compile(optimizer='adam', loss='mean_squared_error', metrics=[PSNR])
    return SRCNN

    