#!/usr/bin/env python3

""" The class in this module is HMMmvndemis (Hidden Markov Model for multivariate normal distributed emission).
It implements HMM with emission probabilities determined by multivariate normal distributions and based on
sklearn.hmm.GaussianHMM class (see https://scikit-learn.sourceforge.net/stable/modules/hmm.html).

The HMM is a generative probabilistic model, in which a seguence of observable X multivariate variable is generated by
a sequence of internal hidden states Z. The hidden states cannot be observed directly. The transitions between hidden
states are assumed to have the form of a (first-order) Markov chain. They can be specified by the start probability
vector P and a transition probability matrix A. The emission probability of an observable in our case are multivariate
normal distributed (Gaussian) with parameters TETA(i) (means(i)(vector) and covariance matrix(i) are estimated along
training dataset, where i - is an state index.
    The HMM is completely determined by P,A, TETA(i) , i = 0,1, number_of_states-1."""

import sys
import logging
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.pca import PCA_

logger = logging.getLogger(__name__)

PATH_TO_CSV="~/LaLaguna/gelPredictor/dataset_Repository/CAISO_Load_01012020_24042023.csv"

class HMMmvndemis():
    """ This class is initialized by following:
    y - time series of observations for demand electricity
    dt - snapshots for observations.
    n_features - number of observations in the segment(day).
    n_states - number of hidden states.

    The time series (TS) y(t)in our model can be represented as X(i,j), where j=0,1,..,n_features -1, i=0,1,2,..
    X -matrix whose rows are days(segn\ments), and whose columns correspond to intraday measurments, i.e., X(2,5) is
    5th measurment (observation) in 2th day.
    The segment(day) may belong to one from n_states possible states.

    The methods of the class evaluate the  sequence of the segment states that satisfies multivariate normal (Gaussian)
    observations of n_feature dimension according the maximum likelihood criterion.

    """
    def __init__(self,y:np.array=None, dt:np.array = None, n_features:int=16, n_states:int=3, log_folder:Path = None ,
                 chart_folder:Path = None, norm:str='norm', title:str="CaISO"):
        if y is None:
            sys.exit(-1)
        self.n_features=n_features
        self.n_states = n_states
        (n,)=y.shape
        if n<self.n_features:
            sys.exit(-2)
        self.n_samples=int(n/self.n_features)
        self.X=y[:self.n_samples * self.n_features].reshape(self.n_samples, self.n_features)
        self.log_folder=log_folder
        self.chart_folder = chart_folder
        self.log = logger
        self.norm = norm
        self.title = title
        if dt is None:
            self.dt=np.array([str(i) for i in range( self.n_samples * self.n_features)])
        else:
            self.dt=dt
        self.y_mean = np.zeros(self.n_samples, dtype=float)
        self.y_std = np.zeros(self.n_samples, dtype=float)
        self.y_min = np.zeros(self.n_samples, dtype=float)
        self.y_max = np.zeros(self.n_samples, dtype=float)
        self.obs_simple_stat()
        self.obs_norm(y=y)

        self.clusters = DataCluster(num_classes=self.n_states)
        n_iter, n_features_in, self.inertia_, self.state_sequence, self.means = self.clusters.fit(X=self.X)
        self.covar = self.covar_estimation()
        self.states, self.state_counts = np.unique(self.state_sequence, return_counts= True)
        self.pai = None
        self.transition = None
        self.model = None



    def obs_simple_stat(self):
        """ The observations -simple statistics """

        for i in range(self.n_samples):
            self.y_mean[i]=self.X[i,:].mean()
            self.y_std[i] = self.X[i, :].std()
            self.y_min[i] = self.X[i, :].min()
            self.y_max[i] = self.X[i, :].max()
        #     charts & logs
        (n,m)=self.X.shape
        if self.log_folder is None:
            file_stats = Path("{}_per_segment_statistics".format(self.title)).with_suffix(".txt")
        else:
            file_stats = Path(self.log_folder / Path("{}_per_segment_statistics".format(self.title))).with_suffix(".txt")

        with open(file_stats, 'w') as fout:
            msg ="{:^5s} {:^30s} {:<12s} {:<12s} {:<12s} {:<12s}\n".format(" ##  ","Timestamp","Mean","Std",
                                                                                       "Min","Max")
            fout.write(msg)
            for i in range(n):
                msg = "{:>5d} {:<30s} {:<12.6e} {:<12.6e} {:<12.6e} {:<12.6e}\n".format(i,self.dt[i*m], \
                        self.y_mean[i],self.y_std[i], self.y_min[i], self.y_max[i])
                fout.write(msg)
            self.log.info("Simple statistics estimation are put in {}".format(file_stats))

        # Plot the sampled data
        if self.chart_folder is None:
            chart_stats = Path("{}_Statistics_per_Day".format(self.title)).with_suffix(".png")
        else:
            chart_stats = Path(self.log_folder / Path("_Statistics_per_Day".format(self.title))).with_suffix(".png")
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        x=np.array([i for i in range(n)])
        plt.plot(x, self.y_mean, label ="average daily power ", alpha=0.7)
        plt.plot(x, self.y_std,  label ="std daily power",  alpha=0.7)
        plt.plot(x,  self.y_min, label = "min daily power", alpha=0.7)
        plt.plot(x,  self.y_max, label = "max daily power", alpha=0.7)
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(chart_stats)
        plt.close("all")
        self.log.info("Simple statistics estimation charts are put in {}".format(chart_stats))
        return

    def obs_norm(self, y:np.array =None):
        """ The observations in the 2D matrix - normalization (-1.0 ...+1.0) """

        self.minmin, self.maxmax = self.X.min(), self.X.max()

        if self.norm == 'norm':
            self.X=(self.X -self.minmin)/(self.maxmax-self.minmin)
            self.log.info("Normalization -1.0 .. +1.0 : minmin : {:<12.6e} maxmax : {:<12.6e}".format( self.minmin, \
                                                                                                   self.maxmax))
            if self.log_folder is None:
                file_norm = Path("{}_norm_tc".format(self.title)).with_suffix(".txt")
            else:
                file_norm = Path(self.log_folder / Path("{}_norm_tc".format(self.title))).with_suffix(".txt")

            if y is None:
                self.log.error("TS is not passed for logging")
                return
            with open(file_norm,'w') as fout:
                #    "{:^5s} {:^30s} {:<12s} {:<12s} {:<12s} {:<12s}\n"
                msg ="{:^5s} {:^5s} {:^9s} {:^30s} {:<12s} {:<12s}\n".format("Ind", "Day", "Intraday"," Timestamp",\
                                                                             "Value","Norm Value")
                fout.write(msg)
                denominator = self.maxmax-self.minmin
                ynorm = (y - self.minmin) / denominator
                for ind in range(len(y)):
                    day=int(ind/self.n_features)
                    intraday = ind % self.n_features

                    # "{:^5s} {:^30s} {:<12s} {:<12s} {:<12s} {:<12s}\n"
                    msg = "{:>5d} {:>5d} {:>9d} {:<30s} {:<12.6e} {:<12.6e}\n".format(ind, day, intraday, self.dt[ind],\
                                                                                      y[ind], ynorm[ind])
                    fout.write(msg)
                self.log.info("Normalized TS put in {}".format(file_norm))

            if self.chart_folder is None:
                chart_norm = Path("{}_norm_tc".format(self.title)).with_suffix(".png")
                chart_ts = Path("{}_tc".format(self.title)).with_suffix(".png")
            else:
                chart_norm = Path(self.log_folder / Path("{}_norm_tc".format(self.title))).with_suffix(".png")
                chart_ts = Path(self.log_folder / Path("{}_tc".format(self.title))).with_suffix(".png")

            plt.rcParams["figure.figsize"] = [12.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            x = np.array([i for i in range(len(y))])

            plt.plot(x, ynorm, label="Normalized TS values", alpha=0.7)

            plt.legend(loc='best')
            # plt.show()
            plt.savefig(chart_norm)
            plt.close("all")
            self.log.info("Nomalized TS values  are put in {}".format(chart_norm))

            plt.rcParams["figure.figsize"] = [12.50, 3.50]
            plt.rcParams["figure.autolayout"] = True

            plt.plot(x, y, label="TS values", alpha=0.7)

            plt.legend(loc='best')
            # plt.show()
            plt.savefig(chart_ts)
            plt.close("all")
            self.log.info("TS values  are put in {}".format(chart_ts))

        return


    def pai_fit(self):
        """ The initial probabilities (Pai) - maximum likelihood estimation (MLE) """

        (n,)=self.state_counts.shape
        if (n<self.n_states):
            self.log.error("Missing states - exit")
            sys.exit(-3)

        sum_ = self.state_counts.sum()
        self.pai=np.array([float(self.state_counts[i]/sum_) for i in range(self.n_states)], dtype=float)
        if self.log_folder is None:
            file_pai = Path("{}_initial probabilities_Pai".format(self.title)).with_suffix(".txt")
        else:
            file_pai = Path(self.log_folder / Path("{}_initial probabilities_Pai".format(self.title))).with_suffix(".txt")

        with open(file_pai, 'w') as fout:
            for i in range (self.n_states):
                fout.write("{:<3d}  {:<12.6e")
            self.log.info("Initial Probabilities are pu in {}".format(file_pai))
        return

    def transition_fit(self):
        """ The transition matrix - MLE"""

        if len(self.state_sequence) == 0 or len(self.states) == 0:
            # logger.error("{} invalid arguments".format(transitionsMLE.__name__))
            self.log.error("State sequence length :{} Number of States : {} - exit. ".format(len(self.state_sequence), \
                                                                                       len(self.states)))
            return None
        if self.state_counts[self.state_sequence[-1]] == 1 :
            # Note: If some state appears only once one as last item in seqiuence then this state will loss.
            self.log.error("some state appears only once one as last item in seqiuence, state shall loss - exit. ")
            sys.exit(-4)
            # Denominators are counts of state occurence along sequence without last item.
        _, denominators = np.unique(self.state_sequence[:-1], return_counts=True)
        # Note: If some state appears only once one as last item in seqiuence then this state will loss.

        self.transition = np.zeros((len(self.states), len(self.states)), dtype=float)

        for statei in self.states:
            denominator = denominators[statei]
            msg = "State {} : ".format(statei)
            for statej in self.states:
                nominator = 0
                for k in range(len(self.state_sequence) - 1):
                    if self.state_sequence[k] == statei and self.state_sequence[k + 1] == statej:
                        nominator += 1
                self.transition[statei][statej] = round(float(nominator) / float(denominator), 6)
                msg = msg + "{} ".format(self.transition[statei][statej])
            message = f"""{msg}"""
            # logger.info(message)
            self.logTransition()
        return

    def logTransition(self):

        if self.log_folder is None:
            file_trans = Path("{}_transition probabilities".format(self.title)).with_suffix(".txt")
        else:
            file_trans = Path(self.log_folder / Path("{}_transition probabilities_".format(self.title))).with_suffix(".txt")
        msg = "  "
        for i in range(len(self.states)):
            msg = msg + " {:^10d}".format(i)
        msg = msg + "\n"
        with open(file_trans, 'w') as fout:
            fout.write(msg)
            for i in range(len(self.states)):
                msg = "{:>3d}".format(i)
                for j in range(len(self.states)):
                    msg = msg + " {:<12.6f}".format(self.transition[i][j])
                msg = msg + "\n"
                fout.write(msg)
            self.log.info("Transition Probabilities matrix is put in {}".format(file_trans))
        return

    def covar_estimation(self)->np.array:
        """ The covariation matrix - MLE.
        (N,M)=X.shape
        cov=(X.T * X)/(N-1), M * M -matrix
        """

        x_mean=self.X.mean(0)
        X = self.X - self.X.mean(axis=0,keepdims=True)
        return np.cov(X.T)


    def fit(self):

        self.pai_fit()
        self.transition_fit()

        self.model = GaussianHMM(n_components=self.n_states, covariance_type="tied")
        self.model.n_features =self.n_features
        # model.transmat=self.transition
        # model.startprob=self.pai

        self.model.means_=self.means
        self.model.covars_ = self.covar

        self.model.fit(self.X)
        vitterbi1= self.model.decode(self.X)
        self.vitterbi = self.model.predict(self.X)
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        x = np.array([i for i in range(self.n_samples)])
        plt.plot(x, self.vitterbi, label="Decoded sequence states", alpha=0.7)
        plt.plot(x, self.state_sequence, label="Sequence states", alpha=0.7)

        plt.legend(loc='best')
        # plt.show()
        if self.chart_folder is None:
            decode_png = Path("{}_{}_State_Decoded_States".format(self.title,self.n_states)).with_suffix(".png")
        else:
            decode_png = Path(self.chart_folder / Path("{}_{}_State_Decoded_States".format(self.title,self.n_states))).with_suffix(".png")
        plt.savefig(decode_png)
        plt.close("all")
        self.log.info("Decode sequence states chart saved in {}".format(decode_png))


    def chartTransitions(self,X:np.array):
        """ """
        (n_samples,n_components) = X.shape
        # plot model states over time
        fig, ax = plt.subplots()
        ax.plot(self.state_sequence, self.vitterbi)
        ax.set_title('States compared to generated')
        ax.set_xlabel('Generated State')
        ax.set_ylabel('Recovered State')
        # fig.show()
        if self.chart_folder is None:
            recovered_png = Path("{}_Generated_Recovered_{}_States.".format(self.title,self.n_states)).with_suffix(".png")
        else:
            recovered_png = Path(self.chart_folder / Path("{}_Generated_Recovered_{}_States.".format(self.title, \
                                                                                self.n_states))).with_suffix(".png")
        fig.savefig(recovered_png)
        self.log.info("Generated Recoverd States chart saved in {}".format(recovered_png))

        # plot the transition matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.imshow(self.transition, aspect='auto', cmap='spring')
        ax1.set_title('Generated Transition Matrix')
        ax2.imshow(self.transition, aspect='auto', cmap='spring')
        ax2.set_title('Recovered Transition Matrix')
        for ax in (ax1, ax2):
            ax.set_xlabel('State To')
            ax.set_ylabel('State From')

        fig.tight_layout()
        # fig.show()
        if self.chart_folder is None:
            recovered_png = Path("{}_Generated_Recovered_Transitions_{}_States".format(self.title, \
                                                                                self.n_states)).with_suffix(".png")
        else:
            recovered_png = Path(self.chart_folder / Path("{}_Generated_Recovered_Transitions_{}_States".format( \
                self.title,self.n_states))).with_suffix(".png")
        fig.savefig(recovered_png)
        self.log.info("Generated Recoverd Transitions States chart saved in {}".format(recovered_png))

        # Plot the sampled data
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
                mfc="orange", alpha=0.7)

        vtrb_states, vtrb_counts = np.unique(self.vitterbi,return_counts=True)
        (n_states,)=vtrb_states.shape
        mean = np.zeros((n_states, n_components), dtype=float)
        for i in range(n_samples):
            st=self.vitterbi[i]
            for j in range(n_components):
                mean[st][j]=mean[st][j] + X[i][j]
        for i in range(n_states) :
            for j in range(n_components):
                mean[i][j]=mean[i][j]/vtrb_counts[i]
        # Indicate the component numbers
        for i, m in enumerate(mean):
            ax.text(m[0], m[1], 'State %i' % (i + 1),
                    size=17, horizontalalignment='center',
                    bbox=dict(alpha=.7, facecolor='w'))
        ax.legend(loc='best')
        # fig.show()
        if self.chart_folder is None:
            obs_states_png = Path("{}_Observations_{}_States".format(self.title,self.n_states)).with_suffix(".png")
        else:
            obs_states_png = Path(self.chart_folder / Path("{}_Observations_{}_States".format(self.title, \
                                                                self.n_states))).with_suffix(".png")
        fig.savefig(obs_states_png)
        self.log.info("GObservations and  States chart saved in {}".format(obs_states_png))


class DataCluster():

    def __init__(self, num_classes:int = 3, log_folder:Path=None):
        pass
        self.num_classes =num_classes
        self.cluster_labels = None
        self.cluster_centers = None


    def fit(self, X:np.array =None)->(int, int, float, np.array, np.array ):
        pass
        (self.n_samples, self.n_features) =X.shape

        kmeans = KMeans(n_clusters=self.num_classes, init='k-means++', random_state=0, n_init=1).fit(X)
        self.cluster_labels=kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        return kmeans.n_iter_, kmeans.n_features_in_, kmeans.inertia_, kmeans.labels_, kmeans.cluster_centers_




if __name__ == "__main__":
    df = pd.read_csv(PATH_TO_CSV)
    y = df["Load"].values
    dt = df["Date Time"].values
    model = HMMmvndemis(y=y, n_features=288, n_states=2, log_folder = None )
    model.fit()
    print(model.y_min, model.y_max, model.y_mean, model.y_std)

    print(model.state_sequence)

    print(model.means)

    pca=PCA_(n_components=2,log_folder=None,title="2comp")
    pca.fit(X=model.X)
    pca.rpt2log()
    model.chartTransitions(pca.X_pca)