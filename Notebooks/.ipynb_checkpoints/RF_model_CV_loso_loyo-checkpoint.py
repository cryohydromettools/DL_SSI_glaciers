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
#    param_grid = [{'n_estimators': [5, 10, 20, 40], 'max_features': [2, 4, 6, 8]},
#                  {'bootstrap': [False], 'n_estimators': [3, 10], 
#                   'max_features': [2, 3, 4]},]
    param_grid = {'n_estimators': [60, 80, 100, 150, 200], 'max_features': [2, 3, 4],
                   'max_depth': [5, 6, 8, 10], 'min_samples_leaf': [2, 4]},

    forest_reg = RandomForestRegressor()
    grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=10, 
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_iter=50)
    return grid_search

# Load data
cs_file = '../data/SMB_input_all_ERA5.csv'
#cs_file = '../data/SMB_input_four_ERA5.csv'

df = pd.read_csv(cs_file,
   delimiter='\t', index_col=['Date'],
    parse_dates=['Date'], na_values='NAN')
labels = df.drop_duplicates(subset=['Stake'])

loso = LeaveOneGroupOut()
groups = df['Stake'].values

X = (df.drop(['SMB', 'Stake'], axis=1)).to_numpy()
y = (df['SMB'].copy()).to_numpy()

# Leave-One-Stake-Out
i = 1
test_rmse = []
for train_index, test_index in loso.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = RF_model().fit(X_train, y_train)

    test_scores = model.cv_results_['mean_test_score']
    train_scores = model.cv_results_['mean_train_score'] 

    fig, (ax0) = plt.subplots(figsize=(6,3))
    ax0.plot(test_scores, label='test')
    ax0.plot(train_scores, label='train')
    ax0.set_ylabel('Score') #fontsize=25
    ax0.set_xlabel('n_estimators') #fontsize=25
    ax0.legend(loc='best')
    fig.savefig('../RF/LOSO_ERA5/CV_FIG/'+str(i)+'_score_RF_stake.png',dpi = 150,
     bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

    joblib.dump(model, '../RF/LOSO_ERA5/CV/stake_'+str(i)+'_model.h5')
    grid_ytest_p = model.predict(X_test)
    grid_test_mse = mean_squared_error(y_test, grid_ytest_p)
    grid_test_rmse = np.sqrt(grid_test_mse)
    test_rmse.append(grid_test_rmse)
    print('Stake'+ ' ' + str(i))
    i= i+1

test_rmse = pd.DataFrame(test_rmse, columns={'RMSE'}, index=labels.Stake.values)
print(test_rmse)

fig, (ax0) = plt.subplots(figsize=(6,3)) 
test_rmse.plot.bar(ax = ax0)
ax0.set_xlabel('')
ax0.set_ylabel(u'RMSE (m w.e.)') 
ax0.set_ylim(0.0, 1.6)
fig.savefig('../fig/rmse_RF_stakes.png',dpi = 150, bbox_inches = 'tight', 
             pad_inches = 0.1, facecolor='w')


df['Year'] = df.index.year
labels = df.drop_duplicates(subset=['Year'])

loyo = LeaveOneGroupOut()
groups = df['Year'].values

X = (df.drop(['SMB', 'Stake', 'Year'], axis=1)).to_numpy()
y = (df['SMB'].copy()).to_numpy()


# Leave-One-Year-Out
i = 1
test_rmse = []
for train_index, test_index in loso.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = RF_model().fit(X_train, y_train)

    test_scores = model.cv_results_['mean_test_score']
    train_scores = model.cv_results_['mean_train_score'] 

    fig, (ax0) = plt.subplots(figsize=(6,3))
    ax0.plot(test_scores, label='test')
    ax0.plot(train_scores, label='train')
    ax0.set_ylabel('Score') #fontsize=25
    ax0.set_xlabel('n_estimators') #fontsize=25
    ax0.legend(loc='best')
    fig.savefig('../RF/LOYO_ERA5/CV_FIG/'+str(i)+'_score_RF_year.png',dpi = 150,
     bbox_inches = 'tight',pad_inches = 0.1, facecolor='w')

    joblib.dump(model, '../RF/LOYO_ERA5/CV/year_'+str(i)+'_model.h5')
    grid_ytest_p = model.predict(X_test)
    grid_test_mse = mean_squared_error(y_test, grid_ytest_p)
    grid_test_rmse = np.sqrt(grid_test_mse)
    test_rmse.append(grid_test_rmse)
    print('Year'+ ' ' + str(i))
    i= i+1


test_rmse = pd.DataFrame(test_rmse, columns={'RMSE'}, index=labels.Year.values).sort_index()

print(test_rmse)

fig, (ax0) = plt.subplots(figsize=(6,3)) 
test_rmse.plot.bar(ax = ax0)
ax0.set_xlabel('')
ax0.set_ylabel(u'RMSE (m w.e.)') 
ax0.set_ylim(0.0, 1.2)
fig.savefig('../fig/rmse_RF_years.png',dpi = 150, bbox_inches = 'tight', 
             pad_inches = 0.1, facecolor='w')


