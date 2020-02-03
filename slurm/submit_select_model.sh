#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath -s $0))

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
while [[ -v CONDA_DEFAULT_ENV ]]; do
    conda deactivate
done

args=()
get_n_jobs=false
for (( i=1; i<=$#; i++ )); do
    args+=(${!i})
    if [[ ${!i} == "--n-jobs" ]]; then
        get_n_jobs=true
    elif [[ $get_n_jobs == true ]]; then
        n_jobs=${!i}
        get_n_jobs=false
    fi
done

if [[ ! -v n_jobs ]]; then
    n_jobs=64
    args+=("--n-jobs" "$n_jobs")
fi
# one more cpu for parent process
n_jobs=$(($n_jobs+1))

sbatch \
--chdir="$(realpath $SCRIPT_PATH/../)" \
--cpus-per-task=$n_jobs \
--gres=lscratch:20 \
--mem-per-cpu=1536m \
--partition=ccr,norm \
--time=48:00:00 \
$SCRIPT_PATH/run_select_model.sh "${args[@]}"
