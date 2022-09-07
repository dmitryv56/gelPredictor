#!/usr/bin/env bash

#  gelpredictor.sh -script location in a projects tree
# ../gelPredictor
#    ../api
#    ../src
#    ../scripts
#             ../gelpredictor.sh
#

# Getting gelPredictor-software root path
GELPREDICTOR_SOFTWARE_PATH=$(dirname "${BASH_SOURCE[0]}")"/../"
GELPREDICTOR_SOFTWARE_PATH=$(readlink -f "${GELPREDICTOR_SOFTWARE_PATH}")
export GELPREDICTOR_SOFTWARE_PATH
echo "GELPREDICTOR_SOFTWARE_PATH ${GELPREDICTOR_SOFTWARE_PATH}"


echo ""
echo "PYTHONPATH: ${PYTHONPATH}"

echo $PYTHONPATH | grep -q "${GELPREDICTOR_SOFTWARE_PATH}"
# echo "Result: $?"
if [ $? -ne 1 ]; then
  export PYTHONPATH
  echo "I am here 1"
else
  if [ -z "$PYTHONPATH" ]; then
    PYTHONPATH=${GELPREDICTOR_SOFTWARE_PATH}
    echo "I am here 2"
  else
    PYTHONPATH=$PYTHONPATH:${GELPREDICTOR_SOFTWARE_PATH}
    echo "I am here 3"
  fi
  export PYTHONPYTH
fi

echo ""
echo "PYTHONPATH: ${PYTHONPATH}"

echo "==========================================================================="
echo "                                                                           "
echo "==========================================================================="

_get_process_pid() {
    local pattern=${1}
#    echo "(_get_process_pid) pattern: ${pattern}"
#    echo ""
    local out=`ps ax | grep ${pattern} | grep --invert-match "grep ${pattern}"`
#    echo "(_get_process_pid) out: ${out}"
#    echo ""

    if [[ "$out" != "" ]]; then
        out=`echo ${out} | awk '{print $1}'`
        echo -n ${out}
    else
        echo -n 0
    fi
}

is_server_predictor_not_running() {
    local _pid=`_get_process_pid "${MSRVCPRED_SERVER_SERVICE}"`
    echo "status: ${_pid}"
    echo ""
    if [[ ${_pid} -eq 0 ]]; then
        echo "server predictor is not running now"
        echo ""
        return 1
    fi
    # echo "server predictor is already running"
    return 0
}

server_service() {
    local is_back=${1}

    #is_server_predictor_not_running || { echo "ERROR" ; echo "Msrvcpred server Service is Running." ; echo "Stop before start!" ; return ; }

     is_server_predictor_not_running
    ret_code=$?
    echo "ret_code: ${ret_code}"
    if [ ${ret_code} -eq 1 ]; then
      echo ""
      echo "ERROR: Msrvcpred server Service is Running.  Stop before start!"
      echo ""
      return
    fi


    echo ${is_back}
    if [[ "${is_back}" == "run_back" ]]; then
        python3 ${MSRVCPRED_SERVER_SERVICE_PATH} &
        echo $!
    else
        python3 ${MSRVCPRED_SERVER_SERVICE_PATH}
    fi
}

predictor_service() {
  python3 ${STGELPDL_SOFTWARE_PATH}/msrvcpred/src/pred_service.py -m auto -t Imbalance &
  local ret=$!
  echo " Predictor service started... ${ret}"
}