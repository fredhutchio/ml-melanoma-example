import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# modeling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input

# loss functions and optimizer
from tensorflow.keras import losses, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy as bin_ce
from tensorflow.keras.losses import categorical_crossentropy as cat_ce

from process import SHAPES


# useful constants and collections
MODELS = {'i':InceptionV3}
LOSSES = {'bin_ce':bin_ce, 'cat_ce':cat_ce}


def get_clean_model(arch, num_classes, wts):
    '''
        Args:
        - arch: str, which pre-trained architecture to use
        - num_classes: int, number of desired output classes
        - wts: str, starting weights to use
        Return:
        - new keras model with specified settings
        - uses specified pre-trained model with additions for fine-tuning
    '''
    # base model is frozen default with no top
    params = {'weights':wts, 'input_shape':SHAPES[arch], 'include_top':False}
    base_model = MODELS[arch](**params)
    base_model.trainable = False
    
    # top part of model
    base_input = keras.Input(shape=SHAPES[arch])
    head_model = base_model(base_input, training=False)
    head_model = GlobalAveragePooling2D()(head_model)
    pred_layer = None
    
    # variable classification layer
    if num_classes == 2:
        pred_layer = Dense(1, activation='softmax')(head_model)
    else:
        pred_layer = Dense(num_classes, activation='softmax')(head_model)
    
    # combine and finish
    ret = Model(base_input, pred_layer)
    return ret


def compile_model(model, loss):
    '''
        Args:
        - model: keras model 
        - loss: str, indicates loss function to use
        Return: 
        - keras model, compiled as specified
    '''
    # set correct metrics
    metrics = []
    if 'bin' in loss:
        metrics = ['BinaryAccuracy', 'AUC', 'Precision', 'Recall']
    else:
        metrics = ['CategoricalAccuracy']
    
    # compile as specified
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss=LOSSES[loss],
        metrics=metrics
    )
    
    # finish
    return model

