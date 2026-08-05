import argparse
import csv
import json
import os
import subprocess
import sys

import pennylane as qp
from pennylane import numpy as np

import main
from checkpoint import loadCheckpoint

DEFAULT_QUBITS = (3, 4, 5)
DEFAULT_REUPLOADS = (1, 2, 3, 5)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#loss_fn is mse_u + mse_f, two terms sharing every parameter, and nothing in the repo currently checks whether they pull the same way. this matters for reading gradient_variance_results.csv honestly: if a config trains badly it could be a barren plateau (gradient magnitude vanishing) or the two loss terms cancelling each other (gradient magnitude fine, direction contested), and those have opposite fixes - change the ansatz vs. reweight the loss. loss_landscape_slice.py sees the shape of the surface but not which term is pulling where. this measures cos(grad mse_u, grad mse_f): near +1 the terms agree, near 0 they are independent, negative means every step that improves the data fit actively worsens the PDE residual


#mse_u alone: how far predictions sit from the labelled initial/boundary data. no pde_residual anywhere, so this is a single level of autodiff
def dataLoss(model, W1, b1, W_q, W2, b2, tData, xData, uData):
    params = (W1, b1, W_q, W2, b2)
    uPred = np.array([model.network(tData[i, 0], xData[i, 0], params)
                      for i in range(len(tData))])
    return np.mean((uPred - uData[:, 0]) ** 2)


#mse_f alone, matching pdeLoss() in gradient_variance.py so the two scripts are measuring the same object. params come in unpacked as separate positional args for the same reason as there - qp.grad returns an empty gradient for a bundled tuple
def residualLoss(model, W1, b1, W_q, W2, b2, tBatch, xBatch):
    params = (W1, b1, W_q, W2, b2)
    residuals = np.array([model.pde_residual(tBatch[i, 0], xBatch[i, 0], params)
                          for i in range(len(tBatch))])
    return np.mean(residuals ** 2)


#flattens the 5-array gradient tuple into one vector so the two terms can be compared as directions in the same parameter space
def flatten(gradTuple):
    return np.concatenate([np.array(g).reshape(-1) for g in gradTuple])


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


#reports the conflict over the whole parameter vector and, separately, over W_q alone. the split matters: the classical pre/post layers can absorb a disagreement that the quantum layer cannot, so a whole-vector cosine near zero with a strongly negative W_q cosine says the contested parameters are exactly the ones the quantum circuit owns
def gradientConflict(nQubits, nReuploads, checkpointPath=None, nData=20, nFBatch=5, seed=0):
    np.random.seed(seed)
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)

    tData, xData, uData, tF, xF = main.make_training_data()
    tData, xData, uData = tData[:nData], xData[:nData], uData[:nData]
    tBatch, xBatch = tF[:nFBatch], xF[:nFBatch]

    if checkpointPath:
        params, _ = loadCheckpoint(checkpointPath)
        params = tuple(np.array(p, requires_grad=True) for p in params)
        state = "trained"
    else:
        params = model.init_params()
        state = "random_init"

    argnums = (0, 1, 2, 3, 4)
    dataFn = lambda *p: dataLoss(model, *p, tData, xData, uData)
    residFn = lambda *p: residualLoss(model, *p, tBatch, xBatch)

    gradData = qp.grad(dataFn, argnums=argnums)(*params)
    gradResid = qp.grad(residFn, argnums=argnums)(*params)

    flatData, flatResid = flatten(gradData), flatten(gradResid)
    wqData = np.array(gradData[2]).reshape(-1)
    wqResid = np.array(gradResid[2]).reshape(-1)

    normData = float(np.linalg.norm(flatData))
    normResid = float(np.linalg.norm(flatResid))

    return {
        "n_qubits": nQubits,
        "n_reuploads": nReuploads,
        "state": state,
        "seed": seed,
        "cos_all_params": cosine(flatData, flatResid),
        "cos_wq_only": cosine(wqData, wqResid),
        "grad_norm_data": normData,
        "grad_norm_residual": normResid,
        #which term dominates the update. far from 1 means the smaller term is effectively
        #ignored by the optimizer regardless of whether the directions agree
        "norm_ratio_data_over_residual": normData / normResid if normResid > 0 else float("inf"),
    }


#same subprocess isolation as gradient_variance.py, and for the same reason: residualLoss goes through pde_residual, which is already two levels of autodiff, and differentiating it w.r.t. all five parameter arrays makes three. looping the grid in one long-lived process leaks device state between configs
def runSingleConfigSubprocess(nQubits, nReuploads, checkpointPath, nData, nFBatch, seed):
    cmd = [sys.executable, os.path.abspath(__file__), "--single",
           "--n-qubits", str(nQubits), "--n-reuploads", str(nReuploads),
           "--n-data", str(nData), "--n-f-batch", str(nFBatch), "--seed", str(seed)]
    if checkpointPath:
        cmd += ["--checkpoint", checkpointPath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"n_qubits={nQubits} n_reuploads={nReuploads} subprocess failed:\n{result.stderr[-2000:]}")
    #json rather than eval on the child's stdout - the row is data, and this way a stray
    #print inside the child is a parse error instead of something that silently executes
    return json.loads(result.stdout.strip().splitlines()[-1])


def runConflictSweep(nQubitsList=DEFAULT_QUBITS, nReuploadsList=DEFAULT_REUPLOADS,
                     checkpointDir=None, nData=20, nFBatch=5, seed=0, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "gradient_conflict.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    fields = ["n_qubits", "n_reuploads", "state", "seed", "cos_all_params", "cos_wq_only",
              "grad_norm_data", "grad_norm_residual", "norm_ratio_data_over_residual"]
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not fileExists:
            writer.writeheader()

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                #if a trained checkpoint exists for this config, measure there too - conflict at a
                #random init and conflict at a converged minimum are different claims, and only the
                #second one explains a training curve that stalled
                paths = [None]
                if checkpointDir:
                    trained = os.path.join(checkpointDir, f"q{nQubits}_r{nReuploads}_s{seed}.pkl")
                    if os.path.exists(trained):
                        paths.append(trained)

                for path in paths:
                    try:
                        row = runSingleConfigSubprocess(nQubits, nReuploads, path, nData, nFBatch, seed)
                    except RuntimeError as e:
                        print(f"n_qubits={nQubits} n_reuploads={nReuploads} FAILED: {e}", flush=True)
                        continue
                    print(f"n_qubits={nQubits} n_reuploads={nReuploads} {row['state']:11s} "
                          f"cos_all={row['cos_all_params']:+.3f} cos_wq={row['cos_wq_only']:+.3f} "
                          f"norm_ratio={row['norm_ratio_data_over_residual']:.2e}", flush=True)
                    writer.writerow(row)
                    f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="measure whether the data term and the PDE-residual term of loss_fn pull the shared parameters in the same direction")
    parser.add_argument("--single", action="store_true",
                        help="internal: compute one config and print its row, used by the subprocess-isolated sweep driver")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--n-reuploads", type=int)
    parser.add_argument("--checkpoint", default=None,
                        help="optional trained checkpoint; without it the measurement is at a random init")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="sweep_checkpoints/ - any q{n}_r{n}_s{seed}.pkl found there gets measured at its trained minimum as well as at a random init")
    parser.add_argument("--n-data", type=int, default=20)
    parser.add_argument("--n-f-batch", type=int, default=5,
                        help="kept small for the same reason as gradient_variance.py: this loss nests three levels of autodiff")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.single:
        print(json.dumps(gradientConflict(args.n_qubits, args.n_reuploads, checkpointPath=args.checkpoint,
                                          nData=args.n_data, nFBatch=args.n_f_batch, seed=args.seed)))
    else:
        runConflictSweep(checkpointDir=args.checkpoint_dir, nData=args.n_data,
                         nFBatch=args.n_f_batch, seed=args.seed)
