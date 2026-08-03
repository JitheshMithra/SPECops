import argparse
import csv
import os
import time

import psutil
from pennylane import numpy as np

import main
import loop
from checkpoint import loadCheckpoint
from eval import evaluate

DEFAULT_QUBITS = (3, 4, 5)
DEFAULT_REUPLOADS = (1, 2, 3, 5)
DEFAULT_SEEDS = (0, 1, 2) #3 seeds minimum per config
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_checkpoints")

def checkpointPathFor(nQubits, nReuploads, checkpointDir=DEFAULT_CHECKPOINT_DIR, seed=None, tag=None):
    label = tag or f"q{nQubits}_r{nReuploads}"
    if seed is not None:
        label += f"_s{seed}"
    return os.path.join(checkpointDir, f"{label}.pkl")

def paramCount(params):
    return int(sum(p.size for p in params))

#peak_wset is windows' own OS-tracked peak working set size for the whole process - there's no resource.getrusage on this platform and no GPU/torch here for cuda.max_memory_allocated, so this is the practical equivalent; note it's a process-wide cumulative peak (monotonically non-decreasing across the whole sweep), not an isolated per-run figure, so later configs will report at least as much as any earlier one
def peakMemoryMB():
    try:
        return psutil.Process(os.getpid()).memory_info().peak_wset / (1024 ** 2)
    except AttributeError: #peak_wset is windows-only; fall back to current RSS elsewhere
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

#one full train+eval replicate for a given model and seed - the building block both runSweep() (quantum configs) and the classical control comparison reuse, so timing/param-count/memory logging live in exactly one place
def trainAndEvalOnce(model, seed, epochs=100, nFBatch=20, evalNx=64, evalNt=50, checkpointPath=None):
    np.random.seed(seed)

    startTime = time.time()
    params, history = loop.train(epochs=epochs, n_f_batch=nFBatch, model=model, checkpointPath=checkpointPath)
    trainTime = time.time() - startTime

    l2Error, pdeError = evaluate(model, params, nx=evalNx, nt=evalNt)

    return {
        "l2_error": l2Error,
        "pde_residual_error": pdeError,
        "final_train_loss": float(history[-1]),
        "train_time_sec": trainTime,
        "param_count": paramCount(params),
        "peak_memory_mb": peakMemoryMB(),
    }

def meanStd(values):
    values = np.array(values, dtype=float)
    return float(np.mean(values)), float(np.std(values))

SWEEP_FIELDNAMES = ["n_qubits", "n_reuploads", "seed",
                     "l2_error", "pde_residual_error", "final_train_loss",
                     "train_time_sec", "param_count", "peak_memory_mb"]

#reads back whichever (n_qubits, n_reuploads, seed) rows already made it to disk from an earlier (possibly crashed) run, so runSweep can skip redoing seeds that already completed instead of retraining from seed 0 every time
def loadCompletedSeeds(resultsPath):
    completed = set()
    if not os.path.exists(resultsPath) or os.path.getsize(resultsPath) == 0:
        return completed
    with open(resultsPath, newline="") as f:
        for row in csv.DictReader(f):
            completed.add((int(row["n_qubits"]), int(row["n_reuploads"]), int(row["seed"])))
    return completed

#loops over (n_qubits, n_reuploads) combos, trains each one seeds-many times through loop.py's training loop, scores each with eval.py's metrics, and writes+flushes one row per seed (not one aggregated row per config) so a run that dies mid-config only loses the seed it was on, not the whole config, and so a restarted run can skip any (n_qubits, n_reuploads, seed) that's already on disk instead of retraining it; mean/std across seeds is printed to the console per config but isn't persisted here, aggregate later from the per-seed rows; the eval grid here is intentionally coarse (64x50, not the full 256x100) to keep total sweep runtime reasonable - each replicate's trained weights get checkpointed too, so runFullGridEval() can get high-res numbers later on just the configs that matter, without retraining
def runSweep(nQubitsList=DEFAULT_QUBITS, nReuploadsList=DEFAULT_REUPLOADS, seeds=DEFAULT_SEEDS,
             epochs=100, nFBatch=20, evalNx=64, evalNt=50, resultsPath=None, checkpointDir=DEFAULT_CHECKPOINT_DIR):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "sweep_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)
    os.makedirs(checkpointDir, exist_ok=True)

    completedSeeds = loadCompletedSeeds(resultsPath)
    fileExists = os.path.exists(resultsPath) and os.path.getsize(resultsPath) > 0

    with open(resultsPath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
        if not fileExists:
            writer.writeheader()
            f.flush()

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                print(f"training n_qubits={nQubits} n_reuploads={nReuploads} ({len(seeds)} seeds)", flush=True)
                model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)

                runs = []
                for seed in seeds:
                    if (nQubits, nReuploads, seed) in completedSeeds:
                        print(f"  seed={seed} already completed, skipping", flush=True)
                        continue

                    checkpointPath = checkpointPathFor(nQubits, nReuploads, checkpointDir, seed=seed)
                    result = trainAndEvalOnce(model, seed, epochs=epochs, nFBatch=nFBatch,
                                               evalNx=evalNx, evalNt=evalNt, checkpointPath=checkpointPath)
                    runs.append(result)
                    print(f"  seed={seed} l2={result['l2_error']:.4f} pdeErr={result['pde_residual_error']:.6f} "
                          f"trainLoss={result['final_train_loss']:.6f} time={result['train_time_sec']:.1f}s", flush=True)

                    writer.writerow({
                        "n_qubits": nQubits, "n_reuploads": nReuploads, "seed": seed,
                        "l2_error": result["l2_error"], "pde_residual_error": result["pde_residual_error"],
                        "final_train_loss": result["final_train_loss"], "train_time_sec": result["train_time_sec"],
                        "param_count": result["param_count"], "peak_memory_mb": result["peak_memory_mb"],
                    })
                    f.flush() #keep partial progress on disk in case a later seed or config crashes or gets killed

                if runs: #only configs with at least one freshly-trained seed have anything new to summarize
                    l2Mean, l2Std = meanStd([r["l2_error"] for r in runs])
                    pdeMean, pdeStd = meanStd([r["pde_residual_error"] for r in runs])
                    print(f"n_qubits={nQubits} n_reuploads={nReuploads} l2={l2Mean:.4f}+/-{l2Std:.4f} "
                          f"pdeErr={pdeMean:.6f}+/-{pdeStd:.6f} (this run's {len(runs)} seed(s) only)", flush=True)

#re-scores one already-swept config's checkpoint at the full 256x100 resolution - meant for the handful of configs that end up getting reported in detail, not for every config in the sweep, since that grid is too slow to run once per config
def runFullGridEval(nQubits, nReuploads, seed=0, checkpointDir=DEFAULT_CHECKPOINT_DIR, resultsPath=None, nx=256, nt=100):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "sweep_full_grid_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    checkpointPath = checkpointPathFor(nQubits, nReuploads, checkpointDir, seed=seed)
    params, _ = loadCheckpoint(checkpointPath)
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)

    l2Error, pdeError = evaluate(model, params, nx=nx, nt=nt)
    print(f"[full grid] n_qubits={nQubits} n_reuploads={nReuploads} seed={seed} l2={l2Error:.6f} pdeErr={pdeError:.6f}")

    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_reuploads", "seed", "l2_error", "pde_residual_error", "nx", "nt"])
        writer.writerow([nQubits, nReuploads, seed, l2Error, pdeError, nx, nt])

    return l2Error, pdeError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-grid", action="store_true", help="re-eval one already-swept config's checkpoint at full 256x100 resolution instead of running the coarse sweep")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--n-reuploads", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    if args.full_grid:
        if args.n_qubits is None or args.n_reuploads is None:
            parser.error("--full-grid requires --n-qubits and --n-reuploads")
        runFullGridEval(args.n_qubits, args.n_reuploads, seed=args.seed, checkpointDir=args.checkpoint_dir, nx=args.nx, nt=args.nt)
    else:
        runSweep()
