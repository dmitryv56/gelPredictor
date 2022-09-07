#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DataSet(object):
    """

    """
    def __init__(self):
        pass
        self.y = []       # time series (TS)
        self.dt =[]       # TS timestamp labels
        self.n =0          # time series length
        self.n_samples =0  # vector (subseries) size. Like as day data , 144 observations
        self.sampling = 10 * 60  # sec


if  __name__ == "__main__":
    pass