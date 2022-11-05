import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utilities.ML_algorithms import create_RF_model, create_XGB_model, create_ANN_model
from utilities.plot_results import plot_prediction, plot_scores, plot_feature_importance
import numpy as np
import joblib

# Set paths

folder_work   = '../XGB/LOSO_ERA5/GROUP'
path_xgb      = folder_work + '/CV/'
path_xgb_fig  = folder_work + '/CV_FIG/'
path_xgb_figs = folder_work + '/CV_Scatter/'
path_xgb_figf = folder_work + '/CV_feature_imp/'


# Read data

cs_file = '../data/SMB_input_four_ERA5.csv'

df_train = pd.read_csv(cs_file, delimiter='\t', index_col=['Date'], parse_dates=['Date'], na_values='NAN')

df_train.reset_index(inplace=True)

# Order by elevation

df_train = df_train.sort_values(by = ['Elevation', 'Date'], ascending=True)

label_stake = df_train.drop_duplicates(subset=['Elevation']).reset_index()[['Elevation', 'Stake']]

print(label_stake)

# Groups to range

label_stake_mod = label_stake.copy()
label_stake_mod.index = label_stake_mod['Elevation']
range_elev = np.arange(90,470, 30)
range_elev[-1] = range_elev[-1] + 30
print(label_stake_mod)

grupos_ele = []
for i in range(len(range_elev)-1):
    print(i)
    grupos_ele.append(label_stake_mod.loc[range_elev[i]:range_elev[i+1]])


print(df_train)

for i in range(len(grupos_ele)):
    df_train['Stake'].replace(grupos_ele[i]['Stake'].values.tolist(), 'G'+str(i), inplace = True)

label_stake = df_train.drop_duplicates(subset=['Stake']).reset_index()[['Stake']]

# Groups to trani

random_stake1 = np.arange(0,6,1)
random_stake2 = np.sort(np.arange(6,12,1))[::-1]
index_test = np.stack((random_stake1, random_stake2), axis=1)

print(index_test)

df_train.index = df_train['Stake']
print(df_train)
print(label_stake)


# Train n folds

for j in range(len(index_test)):

    data_train_df = df_train.loc[label_stake.drop(index_test[j])['Stake'].values]
    data_test_df  = df_train.loc[label_stake.loc[index_test[j]]['Stake'].values]

    data_train_df.dropna(inplace=True)
    data_test_df.dropna(inplace=True)

    features_to_drop = ['Date', 'Stake', 'Latitude','Longitude', 'SMB']

    df_train_X = (data_train_df.drop(features_to_drop, axis=1)) 
    X_train = (data_train_df.drop(features_to_drop, axis=1)).to_numpy()
    y_train = (data_train_df['SMB'].copy()).to_numpy()
    X_test  = (data_test_df.drop(features_to_drop, axis=1)).to_numpy()
    y_test  = (data_test_df['SMB'].copy()).to_numpy()
    print('Fold:' + str(j+1))
    print('Train:'+ ' ' + str(len(X_train)))
    print('Test:'+ ' ' + str(len(X_test)))
    XGB_model = create_XGB_model()
    model = XGB_model.fit(X_train, y_train)
    joblib.dump(model, path_xgb + str(j+1) +'_year_stake_model.h5')

    test_scores = model.cv_results_['mean_test_score']
    train_scores = model.cv_results_['mean_train_score'] 

    # Save plot CV
    fig = plot_scores(test_scores, train_scores)
    fig.savefig(path_xgb_fig + str(j+1) + '_score_XGB_stakes.png',dpi = 150,
    bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')
     
    # Save scatter plot
    fig = plot_prediction(y_test, XGB_model.predict(X_test), len(y_test), n_toplot=5000 )
    #fig = plot_prediction(y_train, XGB_model.predict(X_train), n_toplot=5000)
    fig.savefig(path_xgb_figs + str(j+1) + '_scatter_XGB_stakes.png',dpi = 150,
    bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

    # Save feature importance plot 
    fig = plot_feature_importance(XGB_model, df_train_X, X_test, y_test)
    fig.savefig(path_xgb_figf + str(j+1) + '_feature_XGB_stakes.png',dpi = 150,
    bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

