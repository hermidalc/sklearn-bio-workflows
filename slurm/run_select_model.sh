#!/bin/bash

export MPLBACKEND=agg
export TMPDIR=/lscratch/$SLURM_JOB_ID

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate sklearn-bio-workflows
cmd=("$SLURM_SUBMIT_DIR/select_model.py" "$@")
echo "${cmd[@]}"
"${cmd[@]}"
