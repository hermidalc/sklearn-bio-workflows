#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
while [[ -v CONDA_DEFAULT_ENV ]]; do
    conda deactivate
done

i=1
SELECT_MODEL_OPTS=()
while [[ $i -le $# ]]; do
    if [[ ${!i} == "--n-jobs="* ]]; then
        N_JOBS=${!i#*=}
        SELECT_MODEL_OPTS+=("--n-jobs=$(($N_JOBS - 1))")
    elif [[ ${!i} == "--n-jobs" ]]; then
        SELECT_MODEL_OPTS+=(${!i})
        i=$((i + 1))
        N_JOBS=${!i}
        SELECT_MODEL_OPTS+=($(($N_JOBS - 1)))
    elif [[ ${!i} == "--sbatch-opts="* ]]; then
        SBATCH_OPTS=${!i#*=}
    elif [[ ${!i} == "--sbatch-opts" ]]; then
        i=$((i + 1))
        SBATCH_OPTS=${!i}
    else
        SELECT_MODEL_OPTS+=(${!i})
    fi
    i=$((i + 1))
done

if [[ ! -v N_JOBS ]]; then
    # most Biowulf nodes have 56 CPUs
    N_JOBS=56
    SELECT_MODEL_OPTS+=("--n-jobs" "$(($N_JOBS - 1))")
fi

SCRIPT_PATH=$(dirname $(realpath -s $0))

SBATCH_CMD="sbatch \
--chdir=$(realpath $SCRIPT_PATH/../) \
--cpus-per-task=$N_JOBS \
$SBATCH_OPTS \
$SCRIPT_PATH/run_select_model.sh ${SELECT_MODEL_OPTS[@]}"
echo $SBATCH_CMD
$SBATCH_CMD
