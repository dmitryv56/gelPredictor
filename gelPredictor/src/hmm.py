#!/usr/bin/env python3

""" HMM operates with original (not normalized) time series."""
import logging
import numpy as np

from sys_util.utils import paiMLE, transitionsMLE, emisMLE

logger = logging.getLogger(__name__)

class hmm(object):

    def __init__(self):
        self.log = logger
        self.state_sequence = None
        self.states = None
        self.d_states= {}
        self.pi = None
        self.A = None
        self.B = None

    def setModel(self):

        if self.states is None:
            self.log.error("The state sequence is not set")
            return
        (n_states,)=self.states.shape
        (n,) = self.state_sequence.shape
        # init pi
        self.pi=np.zeros(n_states, dtype=float)

        # init A
        self.A = np.zeros(shape = (n_states,n_states), dtype=float)

    def fit(self, yy:np.array = None,n_block:int=48):
        (n_states,) = self.states.shape
        (train_sequence_size,) = self.state_sequence.shape
        # init pi
        values, counts = np.unique(self.state_sequence, return_counts = True)

        self.pi = paiMLE(states=self.states, count=counts, train_sequence_size=train_sequence_size)

        self.A = transitionsMLE(state_sequence =self.state_sequence, states=self.states)

        self.B = emisMLE(yy=yy[:train_sequence_size * n_block], n_block=n_block, state_sequence=self.state_sequence.tolist(), states=self.states)

    def one_step_predict(self, last_state:int=0)->(int, float):
        """ new_state =argmax(A(i, last_state)"""

        arg_max =0
        (m,n)=self.A.shape
        curr_max = self.A[0,last_state]
        for i in range(n):
            if curr_max<self.A[i, last_state]:
                arg_max = i
                curr_max = self.A[i, last_state]
        return arg_max, self.A[arg_max,last_state]

        """ Predict next state """
if __name__ == "__main__":
    pass