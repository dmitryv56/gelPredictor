#!/usr/bin/env python3


import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


from deltaConfig import PATH_MAIN_LOG, MAX_LOG_SIZE_BYTES, BACKUP_COUNT, PATH_TO_DATASET
from dayProfile import DayProfile

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)


def main():
    pass
if __name__ == "__main__":
    main()


