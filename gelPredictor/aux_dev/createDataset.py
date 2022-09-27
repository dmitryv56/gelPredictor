#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

DS2020 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2020_Excel.csv"
DS2021 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2021_Excel.csv"
DS2022 = "~/LaLaguna/gelPredictor/dataset_Repository/energy-charts_Electricity_generation_in_Germany_in_2022_Excel.csv"
DS202020212022 = "~/LaLaguna/gelPredictor/dataset_Repository/Electricity_generation_in_Germany_2020_2022.csv"
dt_name = "Date Time"
ts_name = "Wind_offshore_50Hertz"
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

if __name__ == "__main__":
    # crtDS()
    readData()
    pass