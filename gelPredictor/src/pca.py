#!/usr/bin/env python3

import logging
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class PCA_():

    def __init__(self,  n_components:int = None, title:str="pca", log_folder:Path = None, chart_folder:Path=None, dt:np.array=None):
        """ """
        self.log = logger
        self.automatic_choice = True
        self.n_components = None
        if n_components > 0:
            self.automatic_choice = False
            self.n_components = n_components
        self.X_pca = None
        self.n_samples =-1
        self.n_features = -1

        self.log_folder = Path.cwd() if log_folder is None else log_folder
        self.chart_folder = Path.cwd() if chart_folder is None else chart_folder
        self.pca = PCA(n_components =self.n_components )
        self.title = title
        self.dt = dt

    def fit(self,X:np.array = None):
        """   """
        if X is None :
            self.log.error("PCA: no data for fit")
            return

        (self.n_samples, self.n_features) = X.shape
        self.X_pca =self.pca.fit_transform(X)

    def rpt2log(self):
        """ Report to log"""
        if self.n_samples==-1 or self.n_features==-1:
            return
        file_components=Path(self.log_folder / Path("{}_right_singular_vectors".format(self.title))).with_suffix(".txt")
        with open(file_components,'w') as fout:
            fout.write("Principial axes in feature space, representing the directions of max.variance in the data for ")
            fout.write("{} samples, {} features, {} componets.\n".format(self.n_samples, self.n_features, self.n_components))
            fout.write("(the right singular vectors of the centered input data, parallel to its eigenvectors,")
            fout.write("sorted by decreasing explained_variance_).\n\n")

            for i in range(self.n_components):
                msg ="{:>3d} ".format(i)
                for j in range(self.n_features):
                    msg= msg + "{:<12.6f} ".format(self.pca.components_[i][j])
                msg = msg + "\n"
                fout.write(msg)

        file_explaind_var = Path(self.log_folder / Path("{}_explained_variances".format(self.title))).with_suffix(".txt")
        with open(file_explaind_var, 'w') as fout:
            fout.write("The variance explained by each of the selected components uses {} - 1  degrees of freedom.\n".format(self.n_samples))
            fout.write("Percentage of variance explained by each of the selected components\n")
            fout.write("The singular values corresponding to each of the selected components.\n")
            for i in range(self.n_components):
                msg="{:>3d} {:<12.6f} {:<12.6f} {:<12.6f}".format(i,self.pca.explained_variance_[i],
                                                               self.pca.explained_variance_ratio_[i],
                                                               self.pca.singular_values_[i])
                msg=msg+ "\n"
                fout.write(msg)

        file_proj2components = Path(self.log_folder / Path("{}_projections2components".format(self.title))).with_suffix(".csv")
        with open(file_proj2components, 'w') as fout:
            if self.dt is None:
                msg="{:<6s}".format("NN")
            else:
                msg = "{:<6s},{:<30s}".format("NN", "Date Time")
            for j in range(self.n_components):
                msg = msg + ",{:<10s}{:<2d}".format("Component",j)
            msg=msg + "\n"
            fout.write(msg)

            for i in range(self.n_samples):
                if self.dt is None:
                    msg = "{:<6d}".format(i)
                else:
                    msg = "{:<6d},{:<30s}\n".format(i, self.dt[i])
                for j in range(self.n_components):
                    msg = msg + ",{:<12.6e}".format(self.X_pca[i][j])
                msg = msg + "\n"
                fout.write(msg)
        return

if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca=PCA_(n_components=1,title="one_components")
    pca.fit(X=X)
    pca.rpt2log()
    pca = PCA_(n_components=2, title="two_components")
    pca.fit(X=X)
    pca.rpt2log()