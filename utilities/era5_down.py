import xarray as xr
import pandas as pd
import numpy as np

def era5_down(files,lon, lat, elev):
    ds = xr.open_mfdataset(files)
    df = ds.sel(longitude=lon, 
                latitude=lat, 
                method='nearest').to_dataframe()
    g       = 9.80665
    hgt_era = df['z'].values[0]/g
    hgt_aws = elev
    df['t2m'] = df['t2m'].values + (hgt_aws - hgt_era) * -0.009
    df['d2m'] = df['d2m'].values + (hgt_aws - hgt_era) * -0.008
    # relative humidity
    T0 = 273.16 # K
    a1 = 611.21 # Pa
    a3 = 17.502 # K
    a4 = 32.19  # K
    R_dry = 287.0597 # Kg^-1 K^-1
    R_vap = 461.5250 # Kg^-1 K^-1

    T  = df['t2m'].values
    Td = df['d2m'].values
    P  = df['sp'].values
    T_e_sat = a1 * np.exp(a3* ((T - T0)/(T - a4)))
    T_q_sat = ((R_dry/R_vap)*T_e_sat)/(P - (1 - (R_dry/R_vap)) * T_e_sat)
    Td_e_sat = a1 * np.exp(a3* ((Td - T0)/(Td - a4)))
    Td_q_sat = ((R_dry/R_vap)*Td_e_sat)/(P - (1 - (R_dry/R_vap)) * Td_e_sat)
    RH    = 100 * Td_e_sat/T_e_sat
    RH[RH > 100]  = 100.0
    RH[RH <   0]  = 0.0
    df['rh'] = RH
    
    # solar radiation
    df['t2m'] = df['t2m'] - 273.16

    # solar radiation
    SWin    = df['ssrd']/3600
    SWin[SWin < 0] = 0.0
    df['SWin'] = SWin

    # long radiation
    LWin    = df['strd']/3600
    df['LWin'] = LWin

    # pressure
    df['press']    = df['sp']/100
    SLP = df['press'].values / np.power((1 - (0.0065 * hgt_era) / (288.15)), 5.255)
    df['press'] = SLP * np.power((1 - (0.0065 * hgt_aws)/(288.15)), 5.22) 

    # msl
    df['msl']    = df['msl']/100

    # wind speed to 2 from 10 m
    U10 = np.sqrt((df['u10'].values)**2 + (df['v10'].values)**2)
    df['u2'] = U10 * (np.log(2/(2.12*1000))/np.log(10/(2.12*1000)))

    # total precipitation
    tp = df['tp'].values + (hgt_aws - hgt_era) * 0.000005
    tp[tp < 0]  = 0.0
    df['tp'] = tp

    # snowfall
    snowfall = df['sf'].values + (hgt_aws - hgt_era) * 0.000005
    snowfall[snowfall < 0]  = 0.0
    df['sf'] = snowfall
    
    df = df[['SWin','LWin', 't2m','rh','u2','press', 'tp','sf','tcc', 'msl']]
    
    return df
