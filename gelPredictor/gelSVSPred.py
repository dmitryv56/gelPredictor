#!/usr/bin/env python3

"""       Short-Term and Very Short-Term forecasting for renewable electricity loading -
g(reen)e(lectricity)l(oading)S(hort-term)V(ery)S(hort-term)Pred(ictor) - gelSVSPred.

    Discretization of the studied and predicted processes in green renewable power generation is performed with
a period of a few minutes (T).  Accordingly, the very short-term (VST) forecast horizon for this discretization T is
1*T, 2*T, 3*T, 4*T, i.e. tens of minutes. On the contrary, the short-term (ST) forecasting horizon is hours or even
days, and medium-term (MT) horizon a month or a quarter.
    For each forecast horizon VST, ST, MT different models are used and possibly aggregation of historical observation
time series (TS). G(reen)E(lectricity)L(oading) Predictor is a tool for VST and ST forecasting based on using
A(rtifical)N(eural)N(et) , H(idden)M(arkov)M(odels), statistical classification  and  image  distingush methods.

 """
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np

from sys_util.parseConfig import  PATH_LOG_FOLDER , PATH_MAIN_LOG , PATH_CHART_LOG , PATH_REPOSITORY , \
MAX_LOG_SIZE_BYTES, BACKUP_COUNT, printInfo
from sys_util.utils import setOutput
from src.flow_control import cntrPath_for_SVSPred

# ------------------------------------------------------------------------------------------------------------------

size_handler = RotatingFileHandler(PATH_MAIN_LOG, mode='a', maxBytes=int(MAX_LOG_SIZE_BYTES),
                                 backupCount=int(BACKUP_COUNT))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)

logger.info(printInfo())
# ------------------------------------------------------------------------------------------------------------------

_, log_folder, chart_log = setOutput(model_repository=PATH_REPOSITORY,
                                     log_folder=PATH_LOG_FOLDER, chart_log=PATH_CHART_LOG)
def main(argc,argv)->int:
    """ Main function """

    ds, X,Y = cntrPath_for_SVSPred()
    with open( Path( PATH_LOG_FOLDER / Path("dataset_info")).with_suffix(".log"), 'w') as fout:
        fout.write( ds._str__())
    (n,)=Y.shape
    (n1,m)=X.shape
    df=pd.DataFrame(np.concatenate((Y.T,X), axis=0))
    try:
        df.to_csv(Path( PATH_LOG_FOLDER / Path("DataForCNN")).with_suffix(".csv"))
    except Exception as err:
        logger.error(err)
    finally:
        pass
    return 0

if __name__ == '__main__':

   try:
       nret = main(None, None)
   except Exception as err:
       logger.error(err)
   finally:
       logger.info('\n\n==============================================================================================')
       logger.info('\n==============================================================================================')
       logger.info('\n\n\n\n\n\n')