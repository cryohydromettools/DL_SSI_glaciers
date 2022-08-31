import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def RF_model():
#    param_grid = [{'n_estimators': [5, 10, 20, 40, 60, 100, 200], 'max_features': [2, 4, 6, 8]},
#                  {'bootstrap': [False], 'n_estimators': [3, 10], 
#                   'max_features': [2, 3, 4]},]

    param_grid = {'n_estimators': [300, 400, 500], 'max_features': [2, 3, 4],
                   'max_depth': [5, 6, 8, 10], 'min_samples_leaf': [20, 25, 30]},

    forest_reg = RandomForestRegressor()
    grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=15, 
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_iter=10)
    return grid_search

n_folds = 10

# Load data
cs_file = '../data/SMB_input_all_ERA5_sea.csv'

#cs_file = '../data/SMB_input_four_ERA5.csv'

# path save figures and RF models 
path_rf = '../RF/LOYSO_ERA5/CV1/'
path_rf_fig = '../RF/LOYSO_ERA5/CV_FIG1/'


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

# 4D array (predictors, n_obs_each_year, n_stake, n_year)

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
    print('Train:'+ ' ' + str(len(X_train)))
    print('Test:'+ ' ' + str(len(X_test)))
    model = RF_model().fit(X_train, y_train)

    test_scores = model.cv_results_['mean_test_score']
    train_scores = model.cv_results_['mean_train_score'] 


    fig, (ax0) = plt.subplots(figsize=(6,3))
    ax0.plot(test_scores, label='test')
    ax0.plot(train_scores, label='train')
    ax0.set_ylabel('Score') #fontsize=25
    ax0.set_xlabel('n_estimators') #fontsize=25
    ax0.legend(loc='best')
    fig.savefig(path_rf_fig + str(j+1)+'_score_RF_year_stake.png',dpi = 150,
     bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

    joblib.dump(model, path_rf + str(j+1)+'_year_stake'+'_model.h5')
    grid_ytest_p = model.predict(X_test)
    grid_test_mse = mean_squared_error(y_test, grid_ytest_p)
    grid_test_rmse = np.sqrt(grid_test_mse)
    test_rmse.append(grid_test_rmse)

test_rmse = pd.DataFrame(test_rmse, columns={'RMSE'})

print(test_rmse)

fig, (ax0) = plt.subplots(figsize=(6,3)) 
test_rmse.plot.bar(ax = ax0)
ax0.set_xlabel('')
ax0.set_ylabel(u'RMSE (m w.e.)') 
ax0.set_ylim(0.0, 1.2)
fig.savefig('../fig/rmse_RF_loyso.png',dpi = 150, bbox_inches = 'tight', 
             pad_inches = 0.1, facecolor='w')
