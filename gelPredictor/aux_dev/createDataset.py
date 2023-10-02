#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import re

DS2020 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2020_Excel.csv"
DS2021 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2021_Excel.csv"
DS2022 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2022_Excel.csv"
DS202020212022 = "~/LaLaguna/gelPredictor/dataset_Repository/Electricity_generation_in_Germany_2020_2022.csv"
DSsolar = "~/LaLaguna/gelPredictor/dataset_Repository/Solar_2021.csv"
DSsolar2021 ="~/LaLaguna/gelPredictor/dataset_Repository/PowerSolar_2021.csv"
DSsolar2021_16 ="~/LaLaguna/gelPredictor/dataset_Repository/PowerSolar_2021_16.csv"
DSsolar2021_5_20 ="~/LaLaguna/gelPredictor/dataset_Repository/PowerSolar_2021_5_20.csv"
DSsolarPlant21012020="~/LaLaguna/gelPredictor/dataset_Repository/SolarPlantPowerGen_21012020.csv"
DSsolarPlant21012020_Light="~/LaLaguna/gelPredictor/dataset_Repository/SolarPlantPowerGen_21012020_Light.csv"
dt_name = "Date Time"
ts_name = "Wind_offshore_50Hertz"
ts_solar_name ="Power_Solar"
ts_power_name="PowerGen"
ts_solar_name_per_day ="Average_Power_Solar_per_day"
DSsolar2021_per_day ="~/LaLaguna/gelPredictor/dataset_Repository/PowerSolar_2021_per_day.csv"

def crtDS():
    df = pd.read_csv(DS2020)
    dt0=df[dt_name].values
    dv0 = df[ts_name].values
    print(dv0.shape)
    del df
    df = pd.read_csv(DS2021)
    dt1 = df[dt_name].values
    dv1 = df[ts_name].values
    print(dv1.shape)
    del df
    df = pd.read_csv(DS2022)
    dt2 = df[dt_name].values
    dv2 = df[ts_name].values
    print(dv2.shape)
    dt=np.concatenate((dt0,dt1,dt2))
    dv = np.concatenate((dv0, dv1, dv2))
    print(dv2.shape)
    del df
    df =pd.DataFrame({dt_name:dt[:], ts_name:dv[:]})
    df.to_csv(DS202020212022)
    pass
def readData():
    df = pd.read_csv(DS202020212022)
    dt = df[dt_name].values
    dv = df[ts_name].values
    print(dv.shape)
    spectr(dv, 1.0/float(15*60))

def spectr(x, fs):
    f, Pxx_den = signal.periodogram(x, fs)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-4, 1e6])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD Wind')

    plt.savefig("spectr.png")
    plt.close("all")

def crtSolarDS():
    df = pd.read_csv(DSsolar)
    dt0=df[dt_name].values
    dv0 = df[ts_solar_name].values
    dt1=[]
    for item in dt0:
        a=item.split(' ')
        c=a[0].split('-')

        d="{}-{}-{} {}".format(c[2],c[1],c[0], a[1])
        dt1.append(d)
    df =pd.DataFrame({dt_name:dt1[:], ts_solar_name:dv0[:]})
    df.to_csv(DSsolar2021)
    pass

def crtSolarDS_16():
    df = pd.read_csv(DSsolarPlant21012020)
    dt0=df[dt_name].values
    dv0 = df[ts_power_name].values
    dt1=[]
    dv1=[]
    n=len(dt0)
    l_light_hours=['08','09','10','11','12','13','14','15','16','17','18','19']
    for i in range(n):
        m=re.search('T(.+?):(.+?):', dt0[i])

        if m.group(1) in l_light_hours:
            dt1.append(dt0[i])
            dv1.append(dv0[i])

    print(len(df))
    df =pd.DataFrame({dt_name:dt1[:], ts_solar_name:dv1[:]})
    df.to_csv(DSsolarPlant21012020_Light)

    print(len(df))
    pass

def crtSolarDSreduce():
    df = pd.read_csv(DSsolar2021)
    dt0=df[dt_name].values
    dv0 = df[ts_solar_name].values
    dt1=[]
    dv1 = []
    for i in range(len(dt0)):
        j=i%24
        if j<5 or j>20:
            continue

        dt1.append(dt0[i])
        dv1.append(dv0[i])
    df =pd.DataFrame({dt_name:dt1[:], ts_solar_name:dv1[:]})
    df.to_csv(DSsolar2021_5_20)
    pass

def spectrAgrSolar(x, fs):
    f, Pxx_den = signal.periodogram(x, fs)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-4, 1e6])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD Agr. Solar Power')

    plt.savefig("spectrSolar.png")
    plt.close("all")


def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in range(lags)]

    return np.array(corr)

def SolarAggrSpect():
    df=pd.read_csv(DSsolar2021)
    dt0 = df[dt_name].values
    dv0 = df[ts_solar_name].values
    dt1 = []
    dv1 = []
    ndays = int(len(dv0)/24)
    n=ndays *24
    if n<len(dv0):
        print("Last {} reduced".format(len(dv0)-n))
    sum_day =0.0
    for i in range(n):
        sum_day=sum_day + dv0[i]

        if i%24 == 0 and i>=24:
            dt1.append(dt0[i-24])
            sum_day=round(sum_day/24,4)
            dv1.append(sum_day)
            sum_day = 0.0
        pass
    pass
    df = pd.DataFrame({dt_name: dt1[:], ts_solar_name: dv1[:]})
    df.to_csv(DSsolar2021_per_day)

    corr = autocorr2(dv1,int(len(dv1)/4))

    print("\n   Autocorelation {}\n".format(ts_solar_name_per_day))
    for i in range(len(corr)):
        print("{:<3d} {:<12.6e}".format(i, corr[i]))

    spectrAgrSolar(dv1, 1.0/(1440*60))
    return

if __name__ == "__main__":
    # crtDS()
    # readData()
    # crtSolarDSreduce()
    # SolarAggrSpect()
    crtSolarDS_16()

    pass