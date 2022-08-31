import sys
import os
import re
import joblib
import pandas as pd
import numpy as np
import dateutil
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from keras.models import load_model
sys.path.append('../')
from utilities.ANN_functions import root_mean_squared_error, r2_keras_loss, r2_keras


dir_ann = '../RF/LOYSO_ERA5/CV/'
models = sorted(os.listdir(dir_ann))

def xgb_predictions(X):
    predictions = {}
    for pkl_file in models:
        file_num = int(re.search(r'\d+', pkl_file).group())
        str_model = os.path.join(dir_ann, pkl_file)
        xgb = joblib.load(str_model)
        y_pred = xgb.predict(X)
        predictions[file_num] = y_pred

    new_df = pd.DataFrame(predictions)
    new_df = new_df[sorted(new_df.columns)]

    return new_df


# Load data
#cs_file = '../data/SMB_input_four_ERA5.csv'
cs_file = '../data/SMB_input_all_ERA5.csv'

df = pd.read_csv(cs_file,
   delimiter='\t', index_col=['Date'],
    parse_dates=['Date'], na_values='NAN')


X = (df.drop(['SMB', 'Stake'], axis=1)).to_numpy()
y = (df['SMB'].copy()).to_numpy()


df_smb = xgb_predictions(X)

df_smb = df_smb.mean(axis=1).to_frame()
df_smb['OBS'] = y
df_smb.rename( columns={0 :'SIM'}, inplace=True )

print(df_smb)

x_era5_RF = df_smb['OBS'].values
y_era5_RF = df_smb['SIM'].values

slope_ERA5_RF, intercept_ERA5_RF, r_value_ERA5_RF, p_value_ERA5_RF, std_err_ERA5_RF = stats.linregress(x_era5_RF,y_era5_RF)

print('r-squared:', r_value_ERA5_RF**2)
print('RMSE:', np.sqrt(mean_squared_error(x_era5_RF,y_era5_RF)))
print('N:', len(y_era5_RF))

# Calculate the point density
xy = np.vstack([x_era5_RF,y_era5_RF])
z_ERA5_RF = gaussian_kde(xy)(xy)


# ANN
dir_ann = '../ANN/LOYSO_ERA5/CV/'
models = sorted(os.listdir(dir_ann))

def xgb_predictions_ann(X):
    predictions = {}
    for pkl_file in models:
        file_num = int(re.search(r'\d+', pkl_file).group())
        str_model = os.path.join(dir_ann, pkl_file)
        xgb = load_model(str_model, custom_objects={"r2_keras": r2_keras, 
                                                    "r2_keras_loss": r2_keras_loss, 
                                                    "root_mean_squared_error": root_mean_squared_error})
        y_pred = xgb.predict(X)
        predictions[file_num] = y_pred[:,0]

    new_df = pd.DataFrame(predictions)
    new_df = new_df[sorted(new_df.columns)]

    return new_df

df_smb = xgb_predictions_ann(X)

df_smb = df_smb.mean(axis=1).to_frame()
df_smb['OBS'] = y
df_smb.rename( columns={0 :'SIM'}, inplace=True )
x_era5_ANN = df_smb['OBS'].values
y_era5_ANN = df_smb['SIM'].values
slope_era5_ANN, intercept_era5_ANN, r_value_era5_ANN,p_value_era5_ANN, std_err_era5_ANN = stats.linregress(x_era5_ANN,y_era5_ANN)

print('r-squared:', r_value_era5_ANN**2)
print('RMSE:', np.sqrt(mean_squared_error(x_era5_ANN,y_era5_ANN)))
print('N:', len(y_era5_ANN))

# Calculate the point density
xy = np.vstack([x_era5_ANN,y_era5_ANN])
z_era5_ANN = gaussian_kde(xy)(xy)


x_int = np.arange(-4,3,0.1)

fig, ((ax, ax1)) = plt.subplots(1,2,figsize=(10,4))

density = ax.scatter(x_era5_RF,y_era5_RF, c=z_ERA5_RF, s=20, vmin=0, vmax=10)
ax.plot(x_int, intercept_ERA5_RF + slope_ERA5_RF * x_int, linewidth=0.8, color='k', linestyle ='-')
ax.axhline(0, linewidth=1, color='grey', linestyle =':')
ax.axvline(0, linewidth=1, color='grey', linestyle =':')
fig.colorbar(density, ax=ax)
ax.set_xlabel('Reference SMB data (m w.e.)')
ax.set_ylabel('SMB modelled with RF (m w.e.)')
ax.set_ylim(-3.0, 2.0)
ax.set_xlim(-3.0, 2.0)
ax.text(-2.8, 1.50,'RMSE'+' = '+str(round(np.sqrt(mean_squared_error(x_era5_RF,y_era5_RF)),2))+' '+'m w.e.', fontsize=12)
ax.text(-2.8, 1.15,'r$^{2}$'+' = '+str(round(r_value_ERA5_RF**2,2)), fontsize=12)
ax.text(-2.8, 0.85,'N'+' = '+str(len(y_era5_ANN)), fontsize=12)
ax.text(1.2, -2.60,'(a)', fontsize=14)

density = ax1.scatter(x_era5_ANN,y_era5_ANN, c=z_era5_ANN, s=20, vmin=0, vmax=10)
ax1.plot(x_int, intercept_era5_ANN + slope_era5_ANN * x_int, linewidth=0.8, color='k', linestyle ='-')
fig.colorbar(density, ax=ax1)
ax1.set_xlabel('Reference SMB data (m w.e.)')
ax1.set_ylabel('SMB modelled with ANN (m w.e.)')
ax1.axhline(0, linewidth=1, color='grey', linestyle =':')
ax1.axvline(0, linewidth=1, color='grey', linestyle =':')
ax1.set_ylim(-3.0, 2.0)
ax1.set_xlim(-3.0, 2.0)
ax1.text(-2.8, 1.50,'RMSE'+' = '+str(round(np.sqrt(mean_squared_error(x_era5_ANN,y_era5_ANN)),2))+' '+'m w.e.', fontsize=12)
ax1.text(-2.8, 1.15,'r$^{2}$'+' = '+str(round(r_value_era5_ANN**2,2)), fontsize=12)
ax1.text(-2.8, 0.85,'N'+' = '+str(len(y_era5_ANN)), fontsize=12)
ax1.text(1.2, -2.60,'(b)', fontsize=14)

fig.savefig('../fig/scatter_plot_RF_ANN_loyso_new.png',dpi = 150, bbox_inches = 'tight', 
             pad_inches = 0.1, facecolor='w')

