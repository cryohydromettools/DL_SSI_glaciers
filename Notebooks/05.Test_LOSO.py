import sys
import os
import re
import joblib
import pandas as pd
sys.path.append('../')
from utilities.plot_results import plot_prediction

# Set paths

folder_work   = '../XGB/LOSO_ERA5/GROUP_test'
path_xgb_figs = folder_work + '/CV_Scatter/'


# Load data
cs_file = '../data/SMB_input_belli_ERA5.csv'

df_test = pd.read_csv(cs_file, delimiter='\t', index_col=['Date'], parse_dates=['Date'], na_values='NAN')

features_to_drop = ['Stake', 'Latitude','Longitude', 'SMB']

X = (df_test.drop(features_to_drop, axis=1)).to_numpy()
y = (df_test['SMB'].copy()).to_numpy()


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


dir_ann = '../XGB/LOSO_ERA5/GROUP/CV/'
models = sorted(os.listdir(dir_ann))

df_smb = xgb_predictions(X)

df_smb = df_smb.mean(axis=1).to_frame()


print(df_smb)

# Save scatter plot
fig = plot_prediction(y, df_smb.values, len(y), n_toplot=5000)
#fig = plot_prediction(y_train, XGB_model.predict(X_train), n_toplot=5000)
fig.savefig(path_xgb_figs + '_scatter_XGB_test.png',dpi = 150,
bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

