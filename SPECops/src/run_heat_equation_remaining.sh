#!/bin/bash
set -e
CONFIGS="3,2 3,3 3,5 4,1 4,2 4,3 4,5 5,1 5,2 5,3 5,5"
for cfg in $CONFIGS; do
    nq="${cfg%,*}"
    nr="${cfg#*,}"
    echo "=== starting q${nq}_r${nr} ===" >> heat_equation_sweep_log.txt
    python heat_equation_sweep.py --n-qubits "$nq" --n-reuploads "$nr" >> heat_equation_sweep_log.txt 2>&1
done
echo "=== all remaining heat-equation configs done ===" >> heat_equation_sweep_log.txt
