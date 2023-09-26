#!/bin/bash
set -xe

DATASET=$1
CURRDIR=$PWD
TIMESTAMP=$(date +%s)
OUTPUT="${CURRDIR}/experiments/${DATASET}-${TIMESTAMP}.zip"

mkdir -p "${CURRDIR}/experiments"
TF_CPP_MIN_LOG_LEVEL=2 python experiment.py --dataset="gs://animal-reid/$DATASET" --output="$OUTPUT" ${@:2}
