#!/bin/bash

SCRIPT_PATH=$(dirname $(realpath -s $0))
export TMPDIR=/lscratch/$SLURM_JOB_ID

conda activate sklearn-bio-workflows
cmd=("$SCRIPT_PATH/../select_model.py" "$@")
echo "${cmd[@]}"
"${cmd[@]}"
