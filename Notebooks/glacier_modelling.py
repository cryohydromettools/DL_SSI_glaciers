import sys
import pandas as pd
import dateutil
import numpy as np
sys.path.append('../')
from utilities.era5_down import era5_down
import glob
import xarray as xr
import os
import re
import joblib
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

# Improved function to sum dataframe columns which contain nan's
def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)

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

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds


t2m = 't2m'
rh2 = 'rh'
ff  = 'u2'
Prec = 'tp'
Snowfall = 'sf'
SWD = 'SWin'
LWD = 'LWin'
Pres = 'press'
fcc  = 'tcc'
msl  = 'msl'


files = sorted(glob.glob('../data/ERA_59_20_day/*.nc'))
print(len(files))

#df_g = gpd.read_file('../PDD_model_chris/data/static/Shapefiles/SSI.shp')
ds   = xr.open_dataset('../../../Collins/mapa/collins_static_2.nc')
mask = ds['MASK'].values
print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")

dso = ds.copy()
dso.coords['time'] = pd.date_range('1959-01-01', '2020-12-31', freq='1M')

smb_dis     = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
accu_dis    = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
t2m_dis     = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
melt_dis    = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
runoff_dis  = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)
MO_dis      = np.full([len(dso.time), len(dso.lat), len(dso.lon)], np.nan)

dir_ann = '../RF/LOYSO_ERA5/CV1/'
models = sorted(os.listdir(dir_ann))

ii = 1
for i in range(len(dso.lat)):

    for j in range(len(dso.lon)):

        if (dso['MASK'][i, j].values == 1):
        	
            print(ii)
            ii = ii + 1
            df_day = era5_down(files, 
                               dso['lon'][j].values, 
                               dso['lat'][i].values,
                               dso['HGT'][i, j].values)
            df_day['t2m_an'] = (df_day[t2m] - df_day[t2m].mean())/df_day[t2m].std()
            df_day['PDD']    = df_day[t2m].where(df_day[t2m] > 0, 0).where(df_day[t2m] <= 0, 1)
            df_month = df_day.resample('1M').agg({t2m:np.mean, rh2:np.mean, ff:np.mean, 
                                      SWD:np.mean, LWD:np.mean, Prec:nansumwrapper,
                                      Snowfall:nansumwrapper, msl:np.mean,
                                      Pres:np.mean, fcc:np.mean, 't2m_an':np.mean,
                                      'PDD':nansumwrapper})
            date_2 = []
            date_2.append(df_month.index[0])
            for tt in df_month.index[0:-1]:
            	date_2.append(tt)

            df_month['Date2'] = date_2
            df_month['Days']  = (df_month.index - df_month['Date2']).dt.days
            df_month['lon']  = dso['lon'][j].values
            df_month['lat']  = dso['lat'][i].values
            df_month['elev'] = dso['HGT'][i, j].values
            df_month_or = df_month[['lat', 'lon', 'elev', 'Days', t2m, rh2, ff, SWD, LWD, Prec, Snowfall, msl, Pres, fcc, 't2m_an', 'PDD']]
            X = df_month_or.to_numpy()
            df_smb = xgb_predictions(X)
            df_smb = df_smb.mean(axis=1).to_frame()
            df_month_or['smb']      = df_smb.values
            df_month_or['melt']     = df_month_or['smb'] - df_month_or[Snowfall]
            df_month_or['melt']     = np.abs(df_month_or['melt'].where(df_month_or['melt'] < 0, 0))
            MO = ((0.003 * df_month_or[t2m].values + 0.52)* df_month_or[Snowfall].values)
            df_month_or['MO']       = MO
            df_month_or['runoff']   = df_month_or['melt'] - df_month_or['MO']
            df_month_or['runoff']   = df_month_or['runoff'].where(df_month_or['runoff'] > 0, 0)
            
            t2m_dis[:, i, j]     = df_month_or[t2m].values
            accu_dis[:, i, j]    = df_month_or[Snowfall].values
            smb_dis[:, i, j]     = df_month_or['smb'].values
            melt_dis[:, i, j]    = df_month_or['melt'].values
            runoff_dis[:, i, j]  = df_month_or['runoff'].values
            MO_dis[:, i, j]      = df_month_or['MO'].values


add_variable_along_timelatlon(dso, t2m_dis, 'T2', '°C', 'Temperature at 2 m')
add_variable_along_timelatlon(dso, accu_dis, 'Accu', 'm w.e.', 'Accumulation')
add_variable_along_timelatlon(dso, smb_dis, 'smb', 'm w.e.', 'Surface Mass Balance')
add_variable_along_timelatlon(dso, melt_dis, 'Melt', 'm w.e.', 'Melt')
add_variable_along_timelatlon(dso, runoff_dis, 'Q', 'm w.e.', 'Runoff')
add_variable_along_timelatlon(dso, MO_dis, 'MO', 'm w.e.', 'Refreezing')
dso.to_netcdf('../data/out/Belli_glacier_new.nc')
