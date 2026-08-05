import argparse
import csv
import importlib
import math
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

#effective_rank.py already measures the entropy rank of the post-quantum-layer OUTPUT matrix, capped at n_qubits - that says how much information the layer emits. this measures the complementary thing in PARAMETER space: the spectrum of the empirical Fisher information F = J^T J / N, where J[i,k] = d u(t_i,x_i) / d param_k. its rank says how many parameter directions actually move the model at all
#
#this is the quantity the parameter-count headline rests on. param_count in sweep_results.csv is nominal - it counts entries in the arrays. if a config carries 71 parameters but its Fisher spectrum has 20 non-negligible directions, then "fewer trainable parameters" and "a smaller model" are different claims and only one of them is true. the normalised effective dimension (Abbas et al. 2021, arXiv:2011.00027) makes that comparable across configs with different param counts, which raw rank is not


#resolves --model to a module the same way eval.py does, then to a per-config model object. main.py exposes build_model(); if a control module only exposes module-level network/init_params, it gets used directly and the qubit/reupload args are ignored
def resolveModel(moduleName, nQubits, nReuploads):
    module = importlib.import_module(moduleName)
    if hasattr(module, "build_model"):
        return module.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
    return module


#one row of the Jacobian per collocation point. differentiating network() rather than pde_residual() on purpose - network is one level of autodiff, so this stays cheap enough to run the whole grid, and the Fisher of the model output is the standard object. differentiating the residual instead would measure the Fisher of the PDE operator applied to the model, which is a different (and much more expensive) thing
def parameterJacobian(model, params, tPoints, xPoints):
    argnums = tuple(range(len(params)))
    rows = []
    for i in range(len(tPoints)):
        fn = lambda *p: model.network(tPoints[i, 0], xPoints[i, 0], p)
        grads = qp.grad(fn, argnums=argnums)(*params)
        rows.append(np.concatenate([np.array(g).reshape(-1) for g in grads]))
    return np.stack(rows)


#entropy-based effective rank, same definition effective_rank.py uses (Roy & Vetterli 2007), so the two numbers are read the same way even though one is over activations and this one is over parameters
def entropyRank(eigenvalues):
    lam = np.clip(np.array(eigenvalues), 0.0, None)
    total = float(np.sum(lam))
    if total <= 0:
        return 0.0
    p = lam / total
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


#normalised effective dimension of Abbas et al. The Fisher is first rescaled to unit average
#eigenvalue (trace = d) - without that the number tracks overall gradient scale rather than the
#number of useful directions, and a config with uniformly tiny gradients would look "low
#dimensional" for the wrong reason. kappa is their gamma*n/(2 pi log n) with n the notional data size
def normalisedEffectiveDimension(eigenvaluesPerInit, nParams, nData=10000, gamma=1.0):
    kappa = gamma * nData / (2 * math.pi * math.log(nData))
    logDets = []
    for eigs in eigenvaluesPerInit:
        lam = np.clip(np.array(eigs), 0.0, None)
        total = float(np.sum(lam))
        if total <= 0:
            continue
        normalised = lam * (nParams / total)
        logDets.append(0.5 * float(np.sum(np.log1p(kappa * normalised))))
    if not logDets:
        return 0.0, 0.0
    logDets = np.array(logDets)
    m = float(np.max(logDets))
    logMean = m + float(np.log(np.mean(np.exp(logDets - m))))
    dEff = 2.0 * logMean / math.log(kappa)
    return float(dEff), float(dEff / nParams) if nParams else 0.0


def fisherSpectrum(nQubits, nReuploads, moduleName="main", checkpointPath=None,
                   nPoints=64, nInits=4, seed=0):
    np.random.seed(seed)
    model = resolveModel(moduleName, nQubits, nReuploads)

    _, _, _, tF, xF = main.make_training_data()
    tPoints, xPoints = tF[:nPoints], xF[:nPoints]

    #a checkpoint pins the measurement to one converged point; without one the effective dimension is
    #averaged over random inits, which is the form the Abbas definition actually calls for
    if checkpointPath:
        loaded, _ = loadCheckpoint(checkpointPath)
        paramSets = [tuple(np.array(p, requires_grad=True) for p in loaded)]
        state = "trained"
    else:
        paramSets = [model.init_params() for _ in range(nInits)]
        state = "random_init"

    eigsPerInit, nParams = [], None
    for params in paramSets:
        jac = parameterJacobian(model, params, tPoints, xPoints)
        nParams = int(jac.shape[1])
        fisher = (jac.T @ jac) / jac.shape[0]
        eigsPerInit.append(np.linalg.eigvalsh(fisher))

    lastEigs = np.sort(eigsPerInit[-1])[::-1]
    positive = lastEigs[lastEigs > 1e-12]
    dEff, dEffNorm = normalisedEffectiveDimension(eigsPerInit, nParams)

    return {
        "n_qubits": nQubits,
        "n_reuploads": nReuploads,
        "model": moduleName,
        "state": state,
        "seed": seed,
        "n_params": nParams,
        "fisher_entropy_rank": entropyRank(lastEigs),
        "fisher_top_eig": float(lastEigs[0]),
        "fisher_trace": float(np.sum(lastEigs)),
        "n_eigs_above_1e-12": int(positive.size),
        #how much of the nominal parameter count is doing anything. this is the column to read next
        #to param_count in sweep_results.csv
        "rank_utilisation": entropyRank(lastEigs) / nParams if nParams else 0.0,
        "effective_dimension": dEff,
        "normalised_effective_dimension": dEffNorm,
        "n_inits_averaged": len(eigsPerInit),
    }


#one level of autodiff only, so this does not have gradient_variance.py's three-deep memory problem.
#kept in a subprocess anyway because the grid loops 12 configs and PennyLane device state leaks
#between them - the same failure that made heat_equation_sweep.py go one config per process
def runSingleConfigSubprocess(nQubits, nReuploads, moduleName, checkpointPath, nPoints, nInits, seed):
    import json
    cmd = [sys.executable, os.path.abspath(__file__), "--single",
           "--n-qubits", str(nQubits), "--n-reuploads", str(nReuploads),
           "--model", moduleName, "--n-points", str(nPoints),
           "--n-inits", str(nInits), "--seed", str(seed)]
    if checkpointPath:
        cmd += ["--checkpoint", checkpointPath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"n_qubits={nQubits} n_reuploads={nReuploads} subprocess failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def runFisherSweep(nQubitsList=DEFAULT_QUBITS, nReuploadsList=DEFAULT_REUPLOADS,
                   moduleName="main", checkpointDir=None, nPoints=64, nInits=4,
                   seed=0, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "fisher_effective_dimension.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    fields = ["n_qubits", "n_reuploads", "model", "state", "seed", "n_params",
              "fisher_entropy_rank", "fisher_top_eig", "fisher_trace", "n_eigs_above_1e-12",
              "rank_utilisation", "effective_dimension", "normalised_effective_dimension",
              "n_inits_averaged"]
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not fileExists:
            writer.writeheader()

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                paths = [None]
                if checkpointDir:
                    trained = os.path.join(checkpointDir, f"q{nQubits}_r{nReuploads}_s{seed}.pkl")
                    if os.path.exists(trained):
                        paths.append(trained)

                for path in paths:
                    try:
                        row = runSingleConfigSubprocess(nQubits, nReuploads, moduleName, path,
                                                        nPoints, nInits, seed)
                    except RuntimeError as e:
                        print(f"n_qubits={nQubits} n_reuploads={nReuploads} FAILED: {e}", flush=True)
                        continue
                    print(f"n_qubits={nQubits} n_reuploads={nReuploads} {row['state']:11s} "
                          f"params={row['n_params']:3d} eff_rank={row['fisher_entropy_rank']:6.2f} "
                          f"utilisation={row['rank_utilisation']:.3f} "
                          f"d_eff_norm={row['normalised_effective_dimension']:.4f}", flush=True)
                    writer.writerow(row)
                    f.flush()


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="parameter-space Fisher spectrum and normalised effective dimension - how much of the nominal parameter count is reachable")
    parser.add_argument("--single", action="store_true",
                        help="internal: compute one config and print its row, used by the subprocess-isolated sweep driver")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--n-reuploads", type=int)
    parser.add_argument("--model", default="main",
                        help="module to import, same convention as eval.py (main / main_classical)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default=None,
                        help="sweep_checkpoints/ - any q{n}_r{n}_s{seed}.pkl found there is measured at its trained minimum as well as at random inits")
    parser.add_argument("--n-points", type=int, default=64,
                        help="collocation points forming the Jacobian; must exceed the parameter count for the Fisher to be full rank in principle")
    parser.add_argument("--n-inits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.single:
        print(json.dumps(fisherSpectrum(args.n_qubits, args.n_reuploads, moduleName=args.model,
                                        checkpointPath=args.checkpoint, nPoints=args.n_points,
                                        nInits=args.n_inits, seed=args.seed)))
    else:
        runFisherSweep(moduleName=args.model, checkpointDir=args.checkpoint_dir,
                       nPoints=args.n_points, nInits=args.n_inits, seed=args.seed)
