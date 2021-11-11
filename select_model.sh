#!/bin/bash

export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export MPLBACKEND=agg
if [ -d "/lscratch/$SLURM_JOB_ID" ]; then
    export TMPDIR=/lscratch/$SLURM_JOB_ID
else
    mkdir -p ~/tmp
    export TMPDIR=~/tmp
fi
# export JOBLIB_TEMP_FOLDER=$TMPDIR
export PYTHONUNBUFFERED=1

python_warnings=(
    'ignore::FutureWarning:sklearn.utils.deprecation'
    'ignore::FutureWarning:rpy2.robjects.pandas2ri'
    'ignore:Optimization did not converge:UserWarning'
    'ignore:Optimization terminated early:UserWarning'
    'ignore:Persisting input arguments took:UserWarning'
    'ignore:Estimator fit failed:RuntimeWarning'
    'ignore:Solver terminated early:UserWarning:sklearn.svm._base'
    'ignore:The max_iter was reached which means the coef_ did not converge:UserWarning:sklearn.linear_model._sag'
    'ignore:No features were selected:UserWarning:sklearn_extensions.feature_selection._base'
)
OIFS="$IFS"
IFS=','
export PYTHONWARNINGS="${python_warnings[*]}"
IFS="$OIFS"

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
while [[ -v CONDA_DEFAULT_ENV ]]; do
    conda deactivate
done
conda activate sklearn-bio-workflows

if [[ -v SLURM_SUBMIT_DIR ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(dirname $(realpath -s $0))
fi

SELECT_MODEL_CMD="$SCRIPT_DIR/select_model.py $@"
echo $SELECT_MODEL_CMD
$SELECT_MODEL_CMD
