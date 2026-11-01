#!/bin/bash
# usage: scripts/unit.sh <unit-name> <logfile> <python args...>   -- survives the launching session
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
name=$1; log=$2; shift 2
# several jobs share 20 cores: an uncapped BLAS pool per process drove the load average to 76
export OMP_NUM_THREADS=${WB_THREADS:-4} OPENBLAS_NUM_THREADS=${WB_THREADS:-4} MKL_NUM_THREADS=${WB_THREADS:-4} NUMEXPR_NUM_THREADS=${WB_THREADS:-4}
exec python "$@" > "$log" 2>&1
