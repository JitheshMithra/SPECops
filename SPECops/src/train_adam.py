import argparse
import csv
import os
import time

import pennylane as qp
from pennylane import numpy as np

import main
import main_classical
from main import make_training_data
from checkpoint import saveCheckpoint
from eval import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#Adam-optimized training loop, model-agnostic like extend_training.trainFrom - built so quantum (main.build_model) and classical (main_classical.build_model) configs can run through the exact same optimizer/epoch/batch settings for a genuinely fair comparison, instead of quantum staying on plain GradientDescentOptimizer while classical switches to Adam. logs L2/residual at every eval checkpoint (not just the final epoch) and checkpoints params at the same cadence, so a crash mid-run (this repo has a known MemoryError-under-contention issue, see extend_training.py) doesn't lose everything the way the earlier GD runs did
def trainAdam(model, epochs, lr, nFBatch, reportEvery, evalEvery, curvePath, checkpointPath):
    params = model.init_params()
    opt = qp.AdamOptimizer(stepsize=lr)

    tData, xData, uData, tF, xF = make_training_data()

    curveFieldnames = ["epoch", "train_loss", "l2_error", "pde_residual_error", "wall_clock_elapsed_sec"]
    with open(curvePath, "w", newline="") as f:
        csv.writer(f).writerow(curveFieldnames)

    startTime = time.time()
    for epoch in range(1, epochs + 1):
        idx = np.random.choice(len(tF), size=nFBatch, replace=False)
        tBatch, xBatch = tF[idx], xF[idx]

        costFn = lambda *p: model.loss_fn(p, tData, xData, uData, tBatch, xBatch)
        params = opt.step(costFn, *params)
        currentLoss = float(costFn(*params))

        if epoch % reportEvery == 0 or epoch == 1:
            elapsed = time.time() - startTime
            print(f"epoch {epoch:5d}  loss={currentLoss:.6f}  elapsed={elapsed:.1f}s", flush=True)

        if epoch % evalEvery == 0 or epoch == epochs:
            l2Error, pdeError = evaluate(model, params, nx=64, nt=50)
            wallClockElapsed = time.time() - startTime
            with open(curvePath, "a", newline="") as f:
                csv.writer(f).writerow([epoch, currentLoss, l2Error, pdeError, wallClockElapsed])
            if checkpointPath:
                saveCheckpoint(checkpointPath, params, model.checkpoint_config())
            print(f"  [eval @ epoch {epoch}] l2={l2Error:.6f} pdeErr={pdeError:.6f}", flush=True)

    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["quantum", "classical_matched"], required=True)
    parser.add_argument("--n-qubits", type=int, default=5)
    parser.add_argument("--n-reuploads", type=int, default=5)
    parser.add_argument("--hidden-width", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-f-batch", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--out-checkpoint", default=None)
    args = parser.parse_args()

    if args.model == "quantum":
        model = main.build_model(n_qubits=args.n_qubits, n_reuploads=args.n_reuploads)
        tag = f"q{args.n_qubits}_r{args.n_reuploads}_adam"
    else:
        model = main_classical.build_model(hidden_width=args.hidden_width)
        tag = "classical_matched_adam"

    print(f"training {tag}: epochs={args.epochs} lr={args.lr} n_f_batch={args.n_f_batch}", flush=True)

    curvePath = os.path.join(RESULTS_DIR, f"{tag}_convergence.csv")
    checkpointPath = args.out_checkpoint

    startTime = time.time()
    params = trainAdam(model, args.epochs, args.lr, args.n_f_batch, args.report_every, args.eval_every,
                        curvePath, checkpointPath)
    trainTime = time.time() - startTime
    print(f"training finished in {trainTime:.1f}s", flush=True)
    print(f"convergence curve saved to {curvePath} (updated live throughout training, not just at the end)", flush=True)

    l2Error, pdeError = evaluate(model, params, nx=64, nt=50)
    print(f"FINAL relative L2 error vs Cole-Hopf: {l2Error:.6f}", flush=True)
    print(f"FINAL PDE residual error: {pdeError:.6f}", flush=True)
