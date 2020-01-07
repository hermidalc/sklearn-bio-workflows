#!/bin/bash

export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export MPLBACKEND=agg
export TMPDIR=/lscratch/$SLURM_JOB_ID

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate sklearn-bio-workflows

cmd=("$SLURM_SUBMIT_DIR/select_model.py" "$@")
echo "${cmd[@]}"
"${cmd[@]}"
