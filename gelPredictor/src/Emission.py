#!/usr/bin/env python

""" The states are set( states list), the state sequence along train path is given (state_sequence list).
Let's consider  a new extended segment (block) representing the average power per day and the time series of daily
power observations of m-size. In our case, its dimension will be m+1
   Knowing that daily observations belong to one of the states, it is possible to estimate the mean vector  and
covariance matrix of m-dimensional observations.
"""

import logging
from pathlib import Path
from math import sqrt
import numpy as np
from numpy.linalg import inv, eigh

from sys_util.utils import exec_time

logger = logging.getLogger(__name__)

DF_ADDITION=10   # Degrees of Freedom addition
REGULARIZATION_FACTOR = 0.05
COND_FEATURE_INDEX = 0

""" This class implements the calculation of the conditional normal distribution of the first componet 
of the vector belongs to the multivariate normal distribution (MND) from the rest.
The parameters of the MND (mean and covariance matrix)  are estimated along the training data path path.
   The training data is an observation vector (yy) whose segments represent multivariate observations. The dimension is 
given by the segment size. Each segment is labeled with a state number. The number of states is known and much smaller 
then the segment size.
   The segments are grouped according to their belonging to the states and the samples of observations are obtained 
with the size Ni by n_block, where Ni is the number of segments belonging to the state 'i'', n_block  is the size 
of the segment.  It is assumed that these are MND-samples with unknown mean vectors (MUi) and unknown covariance 
matrices COVi, where i is state.
   For each i-th state, the mean vectors and covariance matrices are estimated by maximum likelihood method.
    
    Knowing these estimates, estimates of the mean and variance of the conditional distribution of the first coordinate 
of the vector from the rest are calculated. 

   Method fit() forms two vectors are member of class 'mu' and 'sig' the size of them is the size of state_sequence.
    
 """
class Emission():

    def __init__(self, yy: np.array = None, dt:list = [], n_block:int=48, state_sequence: list = [], states:list =[],
                 aver_of_aver:np.array = None, log_folder: Path = None):
        self.log = logger
        self.y = yy
        self.dt = dt
        self.n_block = n_block
        self.m = self.n_block + 1
        self.state_sequence = state_sequence
        self.n =len(self.state_sequence)
        self.states,self.state_cnts=np.unique(self.state_sequence, return_counts=True)
        self.aver_of_aver = aver_of_aver
        self.log_folder=log_folder
    #
        self.ext_segs = []
        self.v_mean = np.zeros(self.m, dtype=float)
        self.mtr_sigma = np.zeros((self.m, self.m), dtype=float)
        self.mu=np.zeros(self.n, dtype=float)
        self.sig = np.ones(self.n, dtype=float)
        self.d_invers={}     # state:inv(Sigma22)

    def fit(self)->np.array:

        """ For each block (segment) the estimate of average is calculated. The extended segments are formed
        (aver0, y0,..,yn_block-1), (aver1, yn_block,..., yn_block+n_block-1),...
        Each segment is treated as a sample vector of multivariate normal distribution with mean (vector) and covariance
         (matrix) are estimated along the train data."""

        (n,)=self.state_sequence.shape
        for n_seg in range(n):
            aux_l =[self.y[i] for i in range(n_seg * self.n_block, (n_seg + 1) * self.n_block)]
            aux_l.insert(0,
                         round(
                             np.sum(self.y[n_seg * self.n_block: (n_seg + 1) * self.n_block]) / float(self.n_block),2
                         ))
            # aux_l.insert(0, self.aver_of_aver[self.state_sequence[n_seg]])
            self.ext_segs.append(aux_l)


        self.estGenPopulation()
        self.logM_N_D( mu=self.v_mean, mtr_Cov=self.mtr_sigma, title="General_Normal_Distribution_Estimation")

        """ The mean estimate for each state 
        for compatibility allocation matrix n_states *2 . The state points on the row. Each row contains a 'mean' and 
        'std' for this state """
        emisDist = np.zeros((len(self.states), 2),  dtype=float)

        for state in self.states:
            v_mean, mtr_Cov = self.estStatePopulation( state=state)
            #  common cov matrix
            # mtr_cov=self.mtr_sigma
            self.logM_N_D(mu=v_mean, mtr_Cov=mtr_Cov, title="State_{}_Normal_Distribution_Estimation".format(state))
            s11, s12, s22, s12_invs22 =self.auxMatrixCalc(v_mean = v_mean, mtr_Cov = mtr_Cov, cond_feature_index = COND_FEATURE_INDEX)


            for n_seg in range(len(self.state_sequence)):
                if self.state_sequence[n_seg] == state:
                    self.sig[n_seg] = s11 - np.dot(s12_invs22, s12)
                    self.mu[n_seg] = v_mean[0] + np.dot(s12_invs22, v_mean[COND_FEATURE_INDEX+1:]
                                                        - self.ext_segs[n_seg][COND_FEATURE_INDEX+1:])
            emisDist[state][0] = v_mean[0] + np.dot(s12_invs22, v_mean[COND_FEATURE_INDEX+1:]
                                                        - self.ext_segs[n_seg][COND_FEATURE_INDEX+1:])
            emisDist[state][1] = sqrt( s11 - np.dot(s12_invs22, s12))

        msg =self.prtEmission()
        self.log.info(msg)


        return emisDist


    def estGenPopulation(self):

        """ The mean estimate for all blocks """
        for a_vec in self.ext_segs:
            for i in range(self.m):
                self.v_mean[i] = self.v_mean[i] + a_vec[i]
        for i in range(self.m):
            self.v_mean[i] = self.v_mean[i]/self.n

        """ The covariance estimate for all blocks """
        X=np.zeros((self.n,self.m), dtype=float)
        for i in range(self.n):
            for j in range(self.m):
                X[i][j]=self.ext_segs[i][j]-self.v_mean[j]
        self.mtr_sigma= np.round((X.T @ X) * 1.0/(self.n-1), 4)
        self.log.info("Check singularity of the covariation matrix estimation for general population\n\n")
        self.checkMatrixSingularity(self.mtr_sigma)
        return

    def checkMatrixSingularity(self, a:np.array)->bool:
        bret = False
        w,v =eigh(a)
        det = 0.0
        trep =0.0
        mineig=np.min(w)
        maxeig=np.max(w)
        w=np.sort(w)
        (m,)=w.shape
        for i in range(m):
            det=det * w[i]
            trep = trep + w[i]
        msg ="Max Eig.Value : {:<10.4f} Min Eig.Value : {:<10.4f} Det. : {:<10.4f} Step : {:<10.4f}\n".format(
            maxeig, mineig, det,trep)
        self.log.info(msg)
        msg="".join("{:<10.4f}  ".format(w[i]) for i in range(m))
        self.log.info("Eigen Values\n{}".format(msg))
        if mineig<1e-07 :
            self.log.error("Ill-conditional matrix")
            bret = True
        return bret

    def estStatePopulation(self, state:int = 0)->(np.array, np.array):

        """ The mean estimate for blocks belongs to state """
        v_mean = np.zeros((self.m), dtype=float)
        mtr_sigma = np.zeros((self.m, self.m), dtype=float)
        for curr_state, a_vec in zip(self.state_sequence, self.ext_segs):
            if curr_state == state:
                for i in range(self.m):
                    v_mean[i] = v_mean[i] + a_vec[i]
        for i in range(self.m):
            v_mean[i] = v_mean[i]/self.state_cnts[state]
        """ The covariance  estimate for blocks belongs to state """
        X=np.zeros((self.state_cnts[state],self.m), dtype=float)
        k=0
        for curr_state, a_vec in zip(self.state_sequence, self.ext_segs):
            if curr_state == state:
                for j in range(self.m):
                    X[k][j]=a_vec[j]- v_mean[j]
                k = k +1

        mtr_sigma= (X.T @ X) * 1.0/(self.state_cnts[state]-1)
        self.log.info("\n\nCheck singularity of the covariation matrix estimation for {} state population\n\n".format(\
            state))
        isSingular = self.checkMatrixSingularity(mtr_sigma)
        if self.state_cnts[state]<self.m+ DF_ADDITION:
            mtr_step=0.0
            for i in range(self.m):
                mtr_step=mtr_step+mtr_sigma[i][i]
            if isSingular:
                for i in range(self.m):
                    mtr_sigma[i][i]=mtr_sigma[i][i] + (1.0 + REGULARIZATION_FACTOR * mtr_step)
                self.log.info("State {}. Cov matrix is ill-conditioned. It was regularized.".format(state))
                self.log.info("\nSingularity status after regularization")
                isSingular = self.checkMatrixSingularity(mtr_sigma)
        return v_mean, mtr_sigma

    """ x=(x1,x2) has multivariate normal distribution (M.N.D.) with mean mu and covariance Sigma.
    x1|x2 has (multivariate)normal distribution with mu1|2 and Sigma1|2, where
       mu1|2 = mu1 + Sigma12 * Inv(Sigma22)* (x2-mu2)
       Sigma1|2 = Sigma11 - Sigma12 *Inv(Sigma22) * Sigma21
       mu.T =(mu1,mu2).T
                [ Sigma11     Sigma12  ]
        Sigma = [                      ]
                [Sigma21       Sigma22 ]
                
        .T -transposition,
        Inv(.) - matrix inversion,
        Sigma21 = Sigms12.T, because Sigma is a covariance matrix,
        x2 - is observations.
    (See in  https://statproofbook.github.io/P/mvn-cond or in Kulback S. Theory Information and Statistics) 
    
    p(x1|x2) = M.N.D.(x1; mu1|2, Sigma1|2)  
    """
    def estCondProbability(self, v_mean:np.array = None, mtr_Cov:np.array=None, x:np.array=None,
                           cond_feature_index:int=0)->(float,float):
        """

        :param v_mean:  - estimation of mean vector (M.N.D) of self.m-size
        :param mtr_Cov: - estimation of covariance matrix (M.N.D.) of (self.m ,self.m) size
        :param x:  - observations vector of self.m-size
        :param cond_feature_index: index of x1 in M.N.D vector(x1,x2). MUST be 0 in current implementations.
        :return: mu - mu1|2 -estimation
        :        sig - Sigma1|2 estimation
        """
        if cond_feature_index>0:
            self.log.error("Cond_feature_index = {} - Not implemented jet. Only 0".format(cond_feature_index))
            return 0.0,1e+06
        cond_feature_index_next = cond_feature_index+1
        s22 = np.array([[mtr_Cov[i][j] for j in range(cond_feature_index_next,self.m)] \
                        for i in range(cond_feature_index_next,self.m)])
        s12 = np.array([mtr_Cov[cond_feature_index][j] for j in range(cond_feature_index_next,self.m)])
        s11 = mtr_Cov[cond_feature_index][cond_feature_index]
        s12_invs22=np.dot(s12,inv(s22))
        sig=s11-np.dot(s12_invs22,s12)
        mu=v_mean[0]+np.dot(s12_invs22, v_mean[cond_feature_index_next:] - x[cond_feature_index_next:])

        return mu, sig

    """ Auxiliry matrix calculations:
    - split mtr_Cov matrix on following submatrix : sigma11 (1 *1 -scalar), 
      Sigma12 (1 *self.m -vector), Sigma22 (self.m * self.m -matrix);;
    - imatrix nversion for Sigma22;
    - calculate  a vector of Sigma12.T * inv(Sigma22).
    """
    def auxMatrixCalc(self, v_mean: np.array = None, mtr_Cov: np.array = None,
                           cond_feature_index: int = 0) -> (float, np.array, float, np.array):
        """

        :param v_mean:
        :param mtr_Cov:
        :param cond_feature_index:
        :return:  Sigma11 -scalar
                  Sigma12 - vector of self.m -size
                  Sigma22 - scalar
                  Sigma12.T * inv(Sigma22) - vector of self.m-size
        """

        if cond_feature_index > 0:
            self.log.error("Cond_feature_index = {} - Not implemented jet. Only 0".format(cond_feature_index))
            return 0.0, 1e+06
        cond_feature_index_next = cond_feature_index + 1
        s22 = np.array([[mtr_Cov[i][j] for j in range(cond_feature_index_next, self.m)] \
                        for i in range(cond_feature_index_next, self.m)])
        s12 = np.array([mtr_Cov[cond_feature_index][j] for j in range(cond_feature_index_next, self.m)])
        s11 = mtr_Cov[cond_feature_index][cond_feature_index]
        s12_invs22 = np.dot(s12, inv(s22))

        return s11, s12, s22, s12_invs22

    def prtEmission(self, emisDist:np.array=None)->str:
        if emisDist is None:
            msg =f""" No emission estimated - ERROR"""
            self.log.error(msg)
            return msg
        msg=""
        for i in range(self.n):
            state =self.state_sequence[i]
            msg = msg + "{:>5d} {:>2d} {:<10.4f} {:<10.4f} {:<30s}\n".format(\
                i, self.state_sequence[i], emisDist,[state][0], emisDist,[state][1], self.dt[i*self.n_block])
        message = f"""
        Emission (approximated by normal distribution)
##### State Mean   Sig       Day (began at) 
{msg}

"""
        return message

    def logM_N_D(self, mu:np.array =None, mtr_Cov:np.array=None,title:str="M.N.D."):

        file_name = "{}.log".format(title)
        if self.log_folder is not None:
            file_name =Path(self.log_folder/ Path(title)).with_suffix(".log)")
        (m,)=mu.shape
        step = 8
        with open(file_name, 'w') as fout:
            fout.write(" Mean vector\n")
            i = 0
            k=round(m/step)
            while (i<k):
                msg =""
                j=i
                while (j<m):
                    msg=msg + "{:>3d} {:<10.4f} ".format(j, mu[j])
                    j=j+k
                msg=msg+"\n"
                fout.write(msg)
                i=i+1

            fout.write("\n\n\n Covariance matrix\n")
            for row in range(m):
                msg = ""
                for col in range(m):
                    msg = msg + "({:>3d},{:>3d}) {:<10.4f} ".format(row,col,mtr_Cov[row][col])
                    if col>0 and (col%step==0) :
                        msg=msg +"\n"
                fout.write(msg)
        return


if __name__ == "__main__":
    pass