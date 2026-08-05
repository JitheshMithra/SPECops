#!/bin/bash
CONFIGS="3,5 4,1 4,2 4,3 4,5 5,1 5,2 5,3 5,5"
LOG=heat_equation_sweep_log.txt
MAX_ATTEMPTS=3
ATTEMPT_OUT=$(mktemp)

#a MemoryError on one config used to kill the whole queue (set -e + bare command); retry that config up to 2 times, only for MemoryError, then move on instead of aborting
for cfg in $CONFIGS; do
    nq="${cfg%,*}"
    nr="${cfg#*,}"
    attempt=1
    configOk=0
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "=== starting q${nq}_r${nr} (attempt ${attempt}/${MAX_ATTEMPTS}) ===" >> "$LOG"
        if python heat_equation_sweep.py --n-qubits "$nq" --n-reuploads "$nr" > "$ATTEMPT_OUT" 2>&1; then
            cat "$ATTEMPT_OUT" >> "$LOG"
            echo "=== q${nq}_r${nr} succeeded on attempt ${attempt}/${MAX_ATTEMPTS} ===" >> "$LOG"
            configOk=1
            break
        fi
        cat "$ATTEMPT_OUT" >> "$LOG"
        if grep -q "MemoryError" "$ATTEMPT_OUT"; then
            echo "=== q${nq}_r${nr} attempt ${attempt}/${MAX_ATTEMPTS} failed with MemoryError, retrying ===" >> "$LOG"
        else
            echo "=== q${nq}_r${nr} attempt ${attempt}/${MAX_ATTEMPTS} failed with a non-MemoryError error, not retrying ===" >> "$LOG"
            break
        fi
        attempt=$((attempt + 1))
    done
    if [ "$configOk" -ne 1 ]; then
        echo "=== q${nq}_r${nr} FAILED after ${attempt} attempt(s), skipping to next config ===" >> "$LOG"
    fi
done

rm -f "$ATTEMPT_OUT"
echo "=== all remaining heat-equation configs done ===" >> "$LOG"
