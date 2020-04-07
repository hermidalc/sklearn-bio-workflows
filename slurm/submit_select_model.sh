#!/bin/bash

shopt -s extglob
USAGE="Usage: $0 --select-model-opts[=]\"<opts>\" --sbatch-opts[=]\"<opts>\""
while getopts ":h-:" OPTCHAR; do
    case "$OPTCHAR" in
        -)
            case "$OPTARG" in
                select-model-opts?(=*))
                    if [[ "$OPTARG" == *"="* ]]; then
                        SELECT_MODEL_OPTS=${OPTARG#*=}
                    else
                        SELECT_MODEL_OPTS="${!OPTIND}"
                        OPTIND=$(($OPTIND + 1))
                    fi
                    ;;
                sbatch-opts?(=*))
                    if [[ "$OPTARG" == *"="* ]]; then
                        SBATCH_OPTS=${OPTARG#*=}
                    else
                        SBATCH_OPTS="${!OPTIND}"
                        OPTIND=$(($OPTIND + 1))
                    fi
                    ;;
                help)
                    echo $USAGE
                    exit 0
                    ;;
                *)
                    if [[ $OPTERR -eq 1 && ${OPTSPEC:0:1} != ":" ]]; then
                        echo "Invalid option --${OPTARG}"
                        echo $USAGE
                        exit 1
                    fi
                    ;;
            esac
            ;;
        h)
            echo $USAGE
            exit 0
            ;;
        *)
            if [[ $OPTERR -eq 1 && ${OPTSPEC:0:1} != ":" ]]; then
                echo "Invalid option -${OPTARG}"
                echo $USAGE
                exit 1
            fi
            ;;
    esac
done
shopt -u extglob
if [[ ! $SELECT_MODEL_OPTS || ! $SBATCH_OPTS ]]; then
    echo $USAGE
    exit 1
fi
SELECT_MODEL_OPTS=($SELECT_MODEL_OPTS)
for (( i=0; i<="${#SELECT_MODEL_OPTS[@]}"; i++ )); do
    if [[ ${SELECT_MODEL_OPTS[i]} == "--n-jobs" ]]; then
        N_JOBS=${SELECT_MODEL_OPTS[i+1]}
        SELECT_MODEL_OPTS[i+1]=$(($N_JOBS - 1))
    fi
done

if [[ ! -v N_JOBS ]]; then
    # most Biowulf nodes have 56 CPUs
    N_JOBS=56
    SELECT_MODEL_OPTS+=("--n-jobs" "$N_JOBS")
fi

SCRIPT_PATH=$(dirname $(realpath -s $0))

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
while [[ -v CONDA_DEFAULT_ENV ]]; do
    conda deactivate
done

sbatch \
--chdir="$(realpath $SCRIPT_PATH/../)" \
--cpus-per-task=$N_JOBS \
$SBATCH_OPTS \
$SCRIPT_PATH/run_select_model.sh "${SELECT_MODEL_OPTS[@]}"
