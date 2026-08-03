import csv
import os

from pennylane import numpy as np

import main
import sweep
from checkpoint import loadCheckpoint
from eval import buildEvalGrid, coleHopfU, relativeL2Error

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SMOOTH_T_MAX = 0.3 #t < this is the "smooth" bucket
SHOCK_T_MIN = 0.5 #t >= this is the "shock" bucket (0.3<=t<0.5 is intentionally excluded as a buffer zone, per the request this stratifies against)

#re-evaluates an already-trained checkpoint's L2 error split by time bucket, instead of retraining - forward passes only (no pde_residual/autodiff), since only L2 vs the Cole-Hopf reference is needed here, which is much cheaper than the full eval.evaluate() that also computes PDE residuals via 3 autodiff calls
def evaluateStratified(model, params, nx=64, nt=50):
    xs, ts = buildEvalGrid(nx, nt)

    smoothPred, smoothRef = [], []
    shockPred, shockRef = [], []
    for t in ts:
        if t < SMOOTH_T_MAX:
            bucketPred, bucketRef = smoothPred, smoothRef
        elif t >= SHOCK_T_MIN:
            bucketPred, bucketRef = shockPred, shockRef
        else:
            continue
        for x in xs:
            bucketPred.append(model.network(t, x, params))
            bucketRef.append(coleHopfU(x, t))

    l2Smooth = relativeL2Error(smoothPred, smoothRef)
    l2Shock = relativeL2Error(shockPred, shockRef)
    return l2Smooth, l2Shock, len(smoothPred), len(shockPred)

def run(nQubitsList=sweep.DEFAULT_QUBITS, nReuploadsList=sweep.DEFAULT_REUPLOADS, seeds=sweep.DEFAULT_SEEDS,
        checkpointDir=sweep.DEFAULT_CHECKPOINT_DIR, nx=64, nt=50, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "sweep_results_stratified.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for nQubits in nQubitsList:
        for nReuploads in nReuploadsList:
            model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
            for seed in seeds:
                checkpointPath = sweep.checkpointPathFor(nQubits, nReuploads, checkpointDir, seed=seed)
                if not os.path.exists(checkpointPath):
                    print(f"MISSING checkpoint for q{nQubits}_r{nReuploads}_s{seed}, skipping")
                    continue
                params, _ = loadCheckpoint(checkpointPath)
                l2Smooth, l2Shock, nSmooth, nShock = evaluateStratified(model, params, nx=nx, nt=nt)
                rows.append({
                    "n_qubits": nQubits, "n_reuploads": nReuploads, "seed": seed,
                    "l2_smooth_t_lt_0.3": l2Smooth, "l2_shock_t_ge_0.5": l2Shock,
                    "n_points_smooth": nSmooth, "n_points_shock": nShock,
                })
                print(f"q{nQubits}_r{nReuploads}_s{seed}: l2_smooth={l2Smooth:.4f} l2_shock={l2Shock:.4f}", flush=True)

    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows

if __name__ == "__main__":
    run()
