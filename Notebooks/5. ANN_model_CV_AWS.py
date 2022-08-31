#if __name__ == '__main__':
    ####

import sys
import pandas as pd
import dateutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)

sys.path.append('../')
from utilities.ANN_functions import root_mean_squared_error, r2_keras_loss, r2_keras
from utilities.ANN_functions import create_loso_model, create_loyo_model, create_lsyso_model

import warnings
warnings.filterwarnings('ignore')
date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)

from dask.distributed import Client
client = Client()

#client = Client(n_workers=2, threads_per_worker=1, memory_limit='1GB')
#client

cs_file = '../data/SMB_input_2011_2015.csv'

df = pd.read_csv(cs_file,
   delimiter='\t', index_col=['Date/Time'],
    parse_dates=['Date/Time'], na_values='NAN',date_parser=date_parser)

X = (df.drop(['Diff', 'Event'], axis=1)).to_numpy()
y = (df['Diff'].copy()).to_numpy()

loso = LeaveOneGroupOut()
groups = df['Event'].values

x, n_features = X.shape

path_ann = '../ANN/LOSO/CV/'
path_ann_fig = '../ANN/LOSO/CV_fig/'

def run_loso():
    # Leave-One-Stake-Out
    i = 1
    test_rmse = []
    for train_index, test_index in loso.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = create_loso_model(n_features, final=False)
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=1000)
        mc = ModelCheckpoint(path_ann + 'stake_'+str(i)+'_model.h5', monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1)

        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                            epochs=2, batch_size = 11, verbose=1, callbacks=[es, mc])

        best_model = load_model(path_ann  + 'stake_'+str(i)+'_model.h5',
                                custom_objects={"r2_keras": r2_keras, 
                                                "r2_keras_loss": r2_keras_loss, 
                                                "root_mean_squared_error": root_mean_squared_error})
        fig, (ax1) = plt.subplots(1,1, figsize=(6,4))
        ax1.plot(history.history['r2_keras'])
        ax1.plot(history.history['val_r2_keras'])
        ax1.set_ylabel('r-square')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='best')
        fig.savefig(path_ann_fig + str(i) + '_r-square_stake'+'.png',dpi = 200, bbox_inches = 'tight', 
                     pad_inches = 0.1, facecolor='w')

        fig, (ax1) = plt.subplots(1,1, figsize=(6,4))
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='best')
        fig.savefig(path_ann_fig + str(i) + '_loss_stake'+'.png',dpi = 200, bbox_inches = 'tight', 
                     pad_inches = 0.1, facecolor='w')

        score = best_model.evaluate(X_test, y_test)

        test_rmse.append(score[0])

        i= i+1
    return test_rmse

#if __name__ == '__main__':

x = client.submit(run_loso)

RMSE_r = pd.DataFrame(data={'RMSE': x.result()})
RMSE_r.to_csv('../out/RMSE_ANN_stakes.csv', sep = ' ', index=False)
print(x.result())
#print(test_rmse)


