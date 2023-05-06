#!/usr/bin/env python3

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

PATH_TO_CSV="~/LaLaguna/gelPredictor/dataset_Repository/CAISO_Load_01012020_24042023.csv"
def main():
    df=pd.read_csv(PATH_TO_CSV)
    y =df["Load"].values
    min_y = y.min()
    ind_min = y.argmin()
    max_y = y.max()
    ind_max = y.argmax()
    aver = y.mean()
    std = y.std()
    n_day=288
    (n1,)=y.shape
    n=int(n1/n_day)
    X=y[:n*n_day].reshape(n,n_day)


    n_states = 5
    n_samples = n
    n_features = n_day
    hmm = GaussianHMM(n_components=n_states)
    # hmm.means_=means

    hmm.fit(X=X)
    Xappr, Z = hmm.sample(n_samples=n_samples)
    pass

if __name__ == "__main__":
    pass
    # n_states=5
    # n_samples=10
    # n_features=2
    # hmm=GaussianHMM(n_components=n_states)
    # x1=np.arange(-0.5,0.5,0.1, dtype=float)
    # x2 = np.arange(-1.0, 1.0, 0.2, dtype=float)
    # (n_samples,)=x1.shape
    # X=np.zeros((n_samples,2), dtype=float)
    #
    # (n_samples,n_features)=X.shape
    # for i in range(n_samples):
    #     X[i][0]=x1[i]
    #     X[i][1]=x2[i]
    # hmm.fit(X=X)
    # X,Z =hmm.sample(n_samples=n_samples)
    # print(Z)
    # print(X)
    main()