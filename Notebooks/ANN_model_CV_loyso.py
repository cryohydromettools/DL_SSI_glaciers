#if __name__ == '__main__':
    ##

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
from utilities.ANN_functions import create_loso_model, create_loyo_model, create_loyso_model

import warnings
warnings.filterwarnings('ignore')
date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)

#from dask.distributed import Client
#client = Client()


n_folds = 10
epochs  = 100

# Load data
cs_file = '../data/SMB_input_all_ERA5_sea.csv'

#cs_file = '../data/SMB_input_four_ERA5.csv'

# path save figures and RF models 
path_ann = '../ANN/LOYSO_ERA5/CV/'
path_ann_fig = '../ANN/LOYSO_ERA5/CV_FIG/'

df = pd.read_csv(cs_file,
   delimiter='\t', index_col=['Date'],
    parse_dates=['Date'], na_values='NAN')
label_stake = df.drop_duplicates(subset=['Stake']).reset_index()[['Stake', 'Latitude', 
                                                                 'Longitude', 'Elevation']]


num_obs = []
for i in label_stake['Stake']:
    stake  = df.loc[lambda df1: df1['Stake'] == i, :].copy().reset_index()
    num_obs.append(stake['Stake'].count())

label_stake['n_obs'] = num_obs

df['Year'] = df.index.year

label_year = df.drop_duplicates(subset=['Year']).reset_index()[['Year']]

num_obs = []
for i in label_year['Year']:
    stake  = df.loc[lambda df1: df1['Year'] == i, :].copy().reset_index()
    num_obs.append(stake['Year'].count())


label_year['n_obs'] = num_obs

label_year.sort_values(by=['Year'], inplace=True)
label_year.reset_index(inplace=True)

stake_obs = np.full([len(label_stake), len(label_year)], np.nan)
stake = 0
for i in label_stake['Stake']:
    stake_sel  = df.loc[lambda df1: df1['Stake'] == i, :].copy().reset_index()
    year = 0
    for j in label_year['Year']:
        stake1  = stake_sel.loc[lambda df1: df1['Year'] == j, :].copy().reset_index()
        stake_obs[stake, year] = len(stake1.SMB.values)
        year = year+1
    stake = stake+1


columns_sel = df.copy()
columns_sel.drop(['Stake', 'Year'], axis=1, inplace = True)

data_y_s = np.full([len(columns_sel.columns), int(stake_obs.max()),
                    len(label_stake), len(label_year)], np.nan)
stake = 0
for i in label_stake['Stake']:#[0:1]:
    stake_sel  = df.loc[lambda df1: df1['Stake'] == i, :].copy().reset_index()
    year = 0
    for j in label_year['Year']:
        stake1  = stake_sel.loc[lambda df1: df1['Year'] == j, :].copy().reset_index()
        SMB = (stake1[columns_sel.columns].values).transpose()
        var = data_y_s[:,:,stake, year]

        if (stake1.shape[0]) != 0:
            for t in range(0,(stake1.shape[0])):
                var[:,t] = SMB[:,t]   
        data_y_s[:, :, stake, year] = var
        year = year+1
    stake = stake+1

prng = np.random.RandomState(2022)
random_years = prng.randint(0, len(label_year), n_folds) # Random year idxs
random_glaciers = prng.randint(0, len(label_stake), n_folds) # Random glacier indexes
lsygo_test_matrixes, lsygo_train_matrixes = [],[]

for i in range(0, n_folds):
    
    test_matrix  = np.zeros((len(label_stake), len(label_year)), dtype=np.int8)
    train_matrix = np.ones((len(label_stake), len(label_year)), dtype=np.int8)

    # Fill test matrix
    test_matrix[random_glaciers[i], :] = 1
    # Fill train matrix
    train_matrix[random_glaciers[i], :] = 0    
    # Fill test matrix
    test_matrix[:, random_years[i]] = 1
    # Fill train matrix
    train_matrix[:, random_years[i]] = 0

    # Add matrixes to folds
    lsygo_test_matrixes.append(test_matrix)
    lsygo_train_matrixes.append(train_matrix)

def run_loyso():
    # Leave-One-Year-Stake-Out
    test_rmse = []
    for j in range(0, n_folds):
        print('Fold'+ ' '+ str(j+1))
        data_train = (data_y_s[:,:,lsygo_train_matrixes[j].astype(bool)])
        data_train_df = pd.DataFrame(data_train[0].flatten(), columns={columns_sel.columns[0]})

        data_test = (data_y_s[:,:,lsygo_test_matrixes[j].astype(bool)])
        data_test_df = pd.DataFrame(data_test[0].flatten(), columns={columns_sel.columns[0]})

        for var in range(1, len(columns_sel.columns)):
            data_train_df[columns_sel.columns[var]] = data_train[var].flatten()
            data_test_df[columns_sel.columns[var]] = data_test[var].flatten()

        data_train_df.dropna(inplace=True)
        data_test_df.dropna(inplace=True)
        
        X_train = (data_train_df.drop(['SMB'], axis=1)).to_numpy()
        y_train = (data_train_df['SMB'].copy()).to_numpy()
        X_test  = (data_test_df.drop(['SMB'], axis=1)).to_numpy()
        y_test  = (data_test_df['SMB'].copy()).to_numpy()

        x, n_features = X_train.shape

        model = create_loyso_model(n_features, final=False)
        es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=1000)
        mc = ModelCheckpoint(path_ann  + str(j+1) + '_year_stake_model.h5', monitor='val_loss', mode='min',
                             save_best_only=True, verbose=1)

        history = model.fit(X_train, y_train, validation_data = (X_test, y_test), 
                            epochs=epochs, batch_size = 19, verbose=1, callbacks=[es, mc])

        best_model = load_model(path_ann  + str(j+1) + '_year_stake_model.h5',
                                custom_objects={"r2_keras": r2_keras, 
                                                "r2_keras_loss": r2_keras_loss, 
                                                "root_mean_squared_error": root_mean_squared_error})
#        fig, (ax1) = plt.subplots(1,1, figsize=(6,4))
#        ax1.plot(history.history['r2_keras'])
#        ax1.plot(history.history['val_r2_keras'])
#        ax1.set_ylabel('r-square')
#        ax1.set_xlabel('epoch')
#        ax1.legend(['train', 'test'], loc='best')
#        fig.savefig(path_ann_fig + str(i) + '_r-square_stake'+'.png',dpi = 200, bbox_inches = 'tight', 
#                     pad_inches = 0.1, facecolor='w')

        fig, (ax1) = plt.subplots(1,1, figsize=(6,4))
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.set_ylim(0,1)
        ax1.legend(['train', 'test'], loc='best')
        fig.savefig(path_ann_fig + str(j+1) + '_loss_year_stake.png',dpi = 200, bbox_inches = 'tight', 
                     pad_inches = 0.1, facecolor='w')

        score = best_model.evaluate(X_test, y_test)

        test_rmse.append(score[0])

    return test_rmse

#x = client.submit(run_loyso)
x = run_loyso()

test_rmse = pd.DataFrame(x, columns={'RMSE'})

print(test_rmse)

fig, (ax0) = plt.subplots(figsize=(6,3)) 
test_rmse.plot.bar(ax = ax0)
ax0.set_xlabel('')
ax0.set_ylabel(u'RMSE (m w.e.)') 
ax0.set_ylim(0.0, 1.2)
fig.savefig('../fig/rmse_ANN_loyso.png',dpi = 150, bbox_inches = 'tight', 
             pad_inches = 0.1, facecolor='w')
