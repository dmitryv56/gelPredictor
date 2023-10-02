#!/usr/bin/env python3

import logging
from datetime import datetime,timedelta, date
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from statsmodels.graphics import tsaplots
import pywt

from deltaConfig import ORDINAL_PROFILE, TYPICAL_PROFILE, LOG_FOLDER_NAME, MAIN_SYSTEM_LOG, CONTINUOUS_WAVELET, \
    NUM_SCALES

logger = logging.getLogger()

class SimpleStat(object):
    """
    """

    def __init__(self, discret:int = 10 * 60, wavelet:object = None, scales: int= NUM_SCALES, log_folder: Path =None,
                 title:str="01/01/2023"):

        self.min  = -1.0
        self.max  = -1.0
        self.mean = -1.0
        self.std  = -1.0
        self.skew = -1.0
        self.discret = discret
        self.wav   = wavelet
        self.scales = self.scales = [i + 1 for i in range(scales)]
        self.title = title

        self.log_folder = log_folder if log_folder is not None else Path().absolute()
        self.log = logger

        self.hist_png = Path(self.log_folder/"Histogram").with_suffix(".png")
        self.hist_log = Path(self.log_folder / "Histogram").with_suffix(".txt")
        self.acf_png = Path(self.log_folder / "ACF").with_suffix(".png")
        self.acf_log = Path(self.log_folder / "ACF").with_suffix(".txt")
        self.psd_png = Path(self.log_folder / "PSD").with_suffix(".png")
        self.psd_log = Path(self.log_folder / "PSD").with_suffix(".txt")
        self.wav_png = Path(self.log_folder / "Wavelet_mexh").with_suffix(".png")
        self.wav_log = Path(self.log_folder / "Scalogram_wav_mexh").with_suffix(".txt")
        return

    def fit(self, data:np.array = None):
        """
        """
        if data is  None:
            self.log.error("")
        (self.n,) = data.shape
        self.min = data.min()
        self.max = data.max()
        self.mean = data.mean()
        self.std = data.std()
        self.skew = stats.skew(data)
        self.hist_plot(data = data, hist_png = self.hist_png, hist_log = self.hist_log)
        self.acf_plot( data = data, acf_png  = self.acf_png,  acf_log  = self.acf_log)
        self.psd_plot( data = data, psd_png  = self.psd_png,  psd_log  = self.psd_log)
        self.wav_plot( data = data, wav_png  = self.wav_png,  wav_log  = self.wav_log, scales=self.scales, wav=self.wav)
        return

    def hist_plot(self, data:np.array = None, hist_png:Path =None, hist_log:Path = None):
        """
        """
        if data is None:
            self.log.error(" No data for histogram")
            return
        if hist_png is None:
            self.log.error(" Invalid file for histogram")
            return

        # the histogram of the data
        (n,)=data.shape
        num_bins = int(n/16) if n<512 else 32
        #bins=np.zeros(num_bins+1, dtype=float)
        fig, ax = plt.subplots()
        hist_values, bins, patches = ax.hist(data, num_bins, density=True)
        # add a 'best fit' line
        y = ((1 / (np.sqrt(2 * np.pi) * self.std)) *
             np.exp(-0.5 * (1 / self.std * (bins - self.mean)) ** 2))
        ax.plot(bins, y, '--')
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability density')
        ax.set_title('Profile Histogram Estimation: '
                     fr'$\mu={self.mean:.0f}$, $\sigma={self.std:.0f}$')

        # Tweak spacing to prevent clipping of ylabel
        fig.tight_layout()
        plt.savefig(hist_png)
        plt.close("all")

        if hist_log is None:
            self.log.error(" Invalid file for histogram log")
            return
        with open(hist_log, 'w') as fout:

            fout.write("{:<3s} {:<12s} {:<12s} {:<12s}\n".format("###","Left Edge", "Right Edge","Value"))
            for i in range(num_bins):
                fout.write("{:<3d} {:<12.6f} {:<12.6f} {:<12.6f}\n".format(i,bins[i], bins[i+1],hist_values[i]))
        return

    def psd_plot(self, data:np.array =None, psd_png:Path =None, psd_log:Path = None):
        """
        """

        if data is None:
            self.log.error(" No data for PSD estimation")
            return
        if psd_png is None:
            self.log.error(" Invalid file for PSD")
            return

        diff = 1.0/self.discret
        for i in range(10):
            if 2**i>self.n:
                break

        NFFT = 2**i
        ax=np.arange(0, self.discret *self.n, self.discret)
        plt.subplot(211)
        plt.plot(ax, data)
        plt.subplot(212)
        psd, freqs = plt.psd(data, NFFT=NFFT, Fs=1.0/self.discret)
        plt.savefig(psd_png)
        plt.close("all")

        if psd_log is None:
            self.log.error(" Invalid file for PSD log")
            return
        with open(psd_log, 'w') as fout:
            fout.write("{:<4s} {:<12s} {:<16s}\n".format("####", "Frequency", "Spectral Density"))
            (n1,)=psd.shape
            for i in range(n1):
                fout.write("{:<4d} {:<12.6f} {:<12.6f}\n".format(i, freqs[i],  psd[i]))

        return

    def acf_plot(self, data:np.array =None, acf_png: Path =None, acf_log: Path = None, ):
        """

        """
        if data is None:
            self.log.error(" No data for ACF estimation")
            return
        if acf_png is None:
            self.log.error(" Invalid file for ACF")
            return
        nlags = int(self.n / 4)
        acf_x = tsaplots.pacf(data, nlags=nlags, alpha=0.05, method=None)
        fig, ax = plt.subplots()
        ax.set_xlabel("Lags")
        ax.set_title("Autocorrelation {}".format(self.title))

        fig = tsaplots.plot_acf(data, ax=ax, lags=int(self.n/4))
        plt.savefig(acf_png)
        plt.close("all")

        if acf_log is None:
            self.log.error(" Invalid file for ACF log")
            return
        with open(acf_log, 'w') as fout:
            fout.write("{:<4s} {:<16s} {:^24s}\n".format("Lag", "Autocorrelation", " Tolerance Range"))

            for i in range(nlags+1):
                fout.write("{:<4d} {:<12.6f} {:<12.6f} {:<12.6f}\n".format(i, acf_x[0][i], acf_x[1][i][0],acf_x[1][i][1]))

        return

    def wav_plot(self, data: np.array = None, scales:int = 16, wav:object = None, wav_png: Path = None, wav_log: Path = None):
        """
        """
        if data is None:
            self.log.error(" No data for Wavelet estimation")
            return
        if wav_png is None:
            self.log.error(" Invalid file for Wavelet image")
            return
        scalogram, freqs = pywt.cwt(data, scales, self.wav)

        simpleScalingImgShow(scalogram=scalogram, title="WAV", wav_png =wav_png, wav_log = wav_log)

        return

def simpleScalingImgShow(scalogram: object = None, title: str = "", wav_png: Path = None, wav_log: Path = None):

    fig, ax = plt.subplots()
    (scale, shift) = scalogram.shape
    extent_lst = [0, shift, 1, scale]
    ax.imshow(scalogram,
              extent=extent_lst,  # extent=[-1, 1, 1, 31],
              cmap='PRGn', aspect='auto',
              vmax=abs(scalogram).max(),
              vmin=-abs(scalogram).max())
    ax.set_title(title)
    ax.set_xlabel("Related Time Series")
    ax.set_ylabel("Scales")
    plt.savefig(wav_png)
    plt.close("all")

    return


class DayProfile(object):
    """

    """

    def __init__(self, discret:int = 10*60, datestamp: datetime.date= None, scales:int = NUM_SCALES,
                 profile:int = ORDINAL_PROFILE , log_folder: Path = LOG_FOLDER_NAME) :

        self.data = None
        self.discret = discret
        self.datestamp = datestamp
        self.profile   = profile

        self.n       =  0
        self.deltaF  =  0.0
        self.nayq    =  0.0
        self.stats   = None
        self.acf_x   = None
        self.psd     = None
        self.freqs   = None

        self.log     = logger
        self.wavelet = CONTINUOUS_WAVELET  #   'mexh'   'cmor1.5-1.0'
        self.wav = pywt.ContinuousWavelet(self.wavelet)
        self.width = self.wav.upper_bound - self.wav.lower_bound
        self.max_wav_len = 0
        self.scales      = scales

        self.date = self.datestamp.strftime("%d_%m_%Y")  # str
        self.log_folder = log_folder if log_folder is not None else Path().absolute()
        self.log_folder = Path(Path(self.log_folder)/Path(self.date))
        self.log_folder.mkdir(parents=True, exist_ok=True)

        return

    def fit(self,data: np.array =None):
        """

        :param data:
        :return:
        """

        if data is None:
            self.log.error(" No data for {} profile ".format(self.date))
            return
        self.data=data
        (self.n,) = self.data.shape
        self.deltaF = 1.0 / (float(self.n) * float(self.discret))
        self.nayq = 1.0 / (2.0 * float(self.discret))
        self.stats = SimpleStat(log_folder = self.log_folder, discret = self.discret, wavelet = self.wav,
                                scales = self.scales).fit(data =self.data)

    def str(self):
        pass





if __name__ == "__main__":
    dp=DayProfile(log_folder=LOG_FOLDER_NAME, datestamp=datetime.now())
    data = np.arange(144, dtype=float)
    (n,) = data.shape
    for i in range(0,n,5):
        data[i]=0.0
    dp.fit(data=data)
