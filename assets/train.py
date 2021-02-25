# IO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import fire
import pickle
import numpy as np
import pandas as pd
import datetime as dt

# data manipulation & model features
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split as tt_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

# model building
from build import get_clean_model, compile_model
from build import LOSSES


# useful constants and collections
DEFAULT_LOG_PATH = './logs/experiments.csv'
LOGGING_COLUMNS = [
    'arch',
    'segged',
    'wts',
    'tuning',
    'num_classes',
    'loss',
    'model_id',
    'model_dir_path'
    'data_id',
    'bs',
    'val_split',
    'epochs'
]


def batch_generator(X, y, batch_size):
    '''
        Args:
        - X: numpy.ndarray, m predictors x n data points
        - y: numpy.ndarray, n target values
        - batch_size: int, how many data points in each returned batch
        Return:
        - batches of X, y as requested by keras.model.fit_generator
    '''
    indices = np.arange(len(X))
    batch = []
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield X[batch], y[batch]
                batch = []

def train(model, model_id, X, y, epochs, val_split, batch_size, mode):
    '''
        Args:
        - model: keras model to train
        - model_id: str, unique identifier for model information storage
        - X: numpy array, attributes
        - y: numpy array, targets
        - epochs: int, number of iterations
        - val_split: float, portion of data to be set aside for validation
        - batch_size: int, number of data points per batch
        - mode: str, indicates whether training or tuning
        Return:
        - str, location of directory with model information
    '''
    # split and report
    X_train, X_val, y_train, y_val = tt_split(X, y, test_size=val_split)
    print('Training shapes: %s %s' % (X_train.shape, y_train.shape))
    print('Validation shapes: %s %s' % (X_val.shape, y_val.shape))
    
    # generators
    train_gen = batch_generator(X_train, y_train, batch_size)
    val_gen = batch_generator(X_val, y_val, batch_size)
    
    # create directory for tensorboard, for this model
    model_tb_path = './logs/tensorboard_%s' % model_id
    os.mkdir(model_tb_path)
    
    # only tensorboard callback for training phase
    callbacks = [TensorBoard(log_dir=model_tb_path)]
    print('TensorBoard directory: %s' % model_tb_path)
    
    # for fine-tuning
    if mode == 'tune':
        early_stop = None
        callbacks.append(early_stop)
    
    # fit
    steps = X_train.shape[0] // batch_size
    val_steps = X_val.shape[0] // batch_size
    print('%d epochs, %d batch size' % (epochs, batch_size))
    print('%d steps per epoch (train), %d (val)' % (steps, val_steps))
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=val_steps
    )
    
    # finish
    model_dir_path = os.path.join('./logs', model_id)
    model.save(model_dir_path)
    
    with open(model_dir_path + '/hist.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return model_dir_path


def main(data_id,
         tuning=False,
         segged=False,
         arch='i',
         wts='imagenet',
         classes=2,
         batch_size=32,
         val_split=0.2,
         epochs=50,
         mode='train'):
    '''
        Args:
        - data_id:
        - tuning:
        - segged:
        - arch:
        - wts:
        - classes:
        - batch_size:
        - val_split:
        - epochs:
        - mode: 
    '''
    # load data
    X = np.load('./data/%s_X.npy' % data_id)
    y = np.load('./data/%s_y.npy' % data_id)
    print('Loaded with shapes %s and %s' % (X.shape, y.shape))
    
    # use classes to determine loss function
    loss = 'bin_ce' if classes == 2 else 'cat_ce'
    print('Inferred %d classes' % classes)
    
    # train/tune and create unique storage id
    model_id = dt.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    model_dir_path = None
    model = get_clean_model(arch, classes, wts)
    model = compile_model(model, loss)
    model_dir_path = train(
        model=model,
        model_id=model_id,
        X=X,
        y=y,
        epochs=epochs,
        val_split=val_split,
        batch_size=batch_size,
        mode=mode
    )

    # log experiment settings
    df_experiments = None
    new_experiment = {
        'arch':arch,
        'segged':segged,
        'wts':wts,
        'tuning':tuning,
        'num_classes':classes,
        'loss':loss,
        'model_id':model_id,
        'model_dir_path':model_dir_path,
        'data_id':data_id,
        'bs':batch_size,
        'val_split':val_split,
        'epochs':epochs
    }
    
    # master file for experiments
    if not os.path.exists(DEFAULT_LOG_PATH):
        df_experiments = pd.DataFrame(columns=LOGGING_COLUMNS)
    else:
        df_experiments = pd.read_csv(DEFAULT_LOG_PATH)
    
    # store and finish
    df_experiments.append(new_experiment)
    df_experiments.to_csv(DEFAULT_LOG_PATH, mode='a', header=False)
    print('Done')
    
                
if __name__ == '__main__':
    # TODO: suppress AutoGraph warning
    fire.Fire(main)