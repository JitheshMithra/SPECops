import argparse
import csv
import os
import time

import pennylane as qp
from pennylane import numpy as np

import main
from main import make_training_data
from checkpoint import loadCheckpoint, saveCheckpoint
from eval import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#continues training from an already-trained checkpoint's params instead of a fresh init - this is loop.train()'s loop body, just seeded with initialParams, so a 100-epoch checkpoint can be pushed to 500+ without redoing the first 100
def trainFrom(model, initialParams, additionalEpochs, lr=0.05, nFBatch=20,
              evalEvery=100, evalNx=64, evalNt=50, startEpoch=0):
    params = initialParams
    opt = qp.GradientDescentOptimizer(stepsize=lr)
    t_data, x_data, u_data, t_f, x_f = make_training_data()

    trainStart = time.time()
    history = []
    evalPoints = []
    for i in range(1, additionalEpochs + 1):
        epoch = startEpoch + i
        idx = np.random.choice(len(t_f), size=nFBatch, replace=False)
        t_f_batch = t_f[idx]
        x_f_batch = x_f[idx]

        cost_fn = lambda *p: model.loss_fn(p, t_data, x_data, u_data, t_f_batch, x_f_batch)
        params = opt.step(cost_fn, *params)
        current_loss = cost_fn(*params)
        history.append(current_loss)

        if epoch % 50 == 0 or i == additionalEpochs:
            print(f"epoch {epoch:4d} loss = {current_loss:.6f}", flush=True)

        if evalEvery and (epoch % evalEvery == 0 or i == additionalEpochs):
            evalStart = time.time()
            l2Error, pdeError = evaluate(model, params, nx=evalNx, nt=evalNt)
            evalTime = time.time() - evalStart
            wallClockElapsed = time.time() - trainStart #cumulative wall-clock since trainFrom started, so time-to-plateau is readable straight from the CSV, not just epoch-to-plateau
            evalPoints.append({"epoch": epoch, "l2_error": l2Error, "pde_residual_error": pdeError,
                                "train_loss": float(current_loss), "wall_clock_elapsed_sec": wallClockElapsed})
            print(f"  [eval @ epoch {epoch}] l2={l2Error:.4f} pdeErr={pdeError:.6f} (eval took {evalTime:.1f}s, wall_clock_elapsed={wallClockElapsed:.1f}s)", flush=True)

    return params, history, evalPoints

def run(nQubits, nReuploads, seed, checkpointPath, additionalEpochs, startEpoch,
        lr=0.05, nFBatch=20, evalEvery=100, evalNx=64, evalNt=50,
        outCsv=None, outCheckpoint=None):
    outCsv = outCsv or os.path.join(RESULTS_DIR, "longer_training_results.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.random.seed(seed) #fresh RNG state for the continuation - not trying to bit-match the original run's stream, just deterministic/reproducible from here on
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
    initialParams, config = loadCheckpoint(checkpointPath)
    print(f"loaded checkpoint {checkpointPath}: config={config}", flush=True)

    startTime = time.time()
    params, history, evalPoints = trainFrom(model, initialParams, additionalEpochs, lr=lr, nFBatch=nFBatch,
                                             evalEvery=evalEvery, evalNx=evalNx, evalNt=evalNt, startEpoch=startEpoch)
    trainTime = time.time() - startTime
    print(f"finished {additionalEpochs} additional epochs (total {startEpoch + additionalEpochs}) in {trainTime:.1f}s", flush=True)

    if outCheckpoint:
        saveCheckpoint(outCheckpoint, params, model.checkpoint_config())
        print(f"saved checkpoint to {outCheckpoint}", flush=True)

    fileExists = os.path.exists(outCsv)
    with open(outCsv, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["model", "n_qubits", "n_reuploads", "seed", "total_epochs", "l2_error", "pde_residual_error", "train_loss", "wall_clock_elapsed_sec"])
        for point in evalPoints:
            writer.writerow(["quantum_pinn", nQubits, nReuploads, seed, point["epoch"], point["l2_error"], point["pde_residual_error"],
                              point["train_loss"], point["wall_clock_elapsed_sec"]])

    return params, history, evalPoints

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--n-reuploads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True, help="existing trained checkpoint to continue from")
    parser.add_argument("--additional-epochs", type=int, required=True)
    parser.add_argument("--start-epoch", type=int, default=100, help="epoch count the input checkpoint already represents")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-f-batch", type=int, default=20)
    parser.add_argument("--eval-nx", type=int, default=64)
    parser.add_argument("--eval-nt", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-checkpoint", default=None)
    args = parser.parse_args()

    run(args.n_qubits, args.n_reuploads, args.seed, args.checkpoint, args.additional_epochs, args.start_epoch,
        lr=args.lr, nFBatch=args.n_f_batch, evalEvery=args.eval_every, evalNx=args.eval_nx, evalNt=args.eval_nt,
        outCsv=args.out, outCheckpoint=args.out_checkpoint)
