import argparse
import csv
import os
import time

from pennylane import numpy as np

import heat_equation as heq
import loop
import sweep

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat_equation_checkpoints")

#trains and evaluates ONE (n_qubits, n_reuploads) config for the heat equation, single seed - CLI-invoked once per config (see run_heat_equation_sweep.sh-style orchestration) rather than looping over all 12 configs inside one process, because looping in-process was building a fresh PennyLane device/qnode per config without releasing the previous one's state, which crashed with a MemoryError partway through the second config even with only one other process running concurrently. One process per config matches every other proven-stable script in this repo (extend_training.py, train_adam.py, shapley_layer_attribution.py all only ever build one device per process lifetime).
def runOne(nQubits, nReuploads, seed=0, epochs=100, nFBatch=20, evalNx=64, evalNt=50,
           resultsPath=None, checkpointDir=CHECKPOINT_DIR):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "heat_equation_sweep.csv")
    os.makedirs(checkpointDir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    print(f"training q{nQubits}_r{nReuploads} (heat equation, seed={seed}, epochs={epochs})", flush=True)
    model = heq.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
    checkpointPath = os.path.join(checkpointDir, f"q{nQubits}_r{nReuploads}_s{seed}.pkl")

    np.random.seed(seed)
    startTime = time.time()
    params, history = loop.train(epochs=epochs, n_f_batch=nFBatch, model=model, checkpointPath=checkpointPath)
    trainTime = time.time() - startTime

    l2Error, pdeError = heq.evaluate(model, params, nx=evalNx, nt=evalNt)
    print(f"q{nQubits}_r{nReuploads}: l2={l2Error:.6f} pdeErr={pdeError:.6f} trainTime={trainTime:.1f}s", flush=True)

    fileExists = os.path.exists(resultsPath) and os.path.getsize(resultsPath) > 0
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_reuploads", "seed", "l2_error", "pde_residual_error", "train_time_sec"])
        writer.writerow([nQubits, nReuploads, seed, l2Error, pdeError, trainTime])

    print(f"appended to {resultsPath}", flush=True)
    return l2Error, pdeError, trainTime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--n-reuploads", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-f-batch", type=int, default=20)
    args = parser.parse_args()

    runOne(args.n_qubits, args.n_reuploads, seed=args.seed, epochs=args.epochs, nFBatch=args.n_f_batch)
