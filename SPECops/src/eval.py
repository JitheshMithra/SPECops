import argparse
import csv
import importlib
import os
import time

from pennylane import numpy as np
from scipy.integrate import quad

import main
from checkpoint import loadCheckpoint

#cole-hopf gives the exact solution to burgers' eq for our IC/BC, but the integrals don't collapse to anything elementary, so we fall back to quad. the exp(-eta^2 / 4*nu*t) term is a gaussian centered at eta=0 with width ~sqrt(nu*t), and nu is tiny here (~0.003), so for small t that gaussian is a needle - a fixed +/-10 window starves quad of samples near the peak and it reports a divergent/roundoff integral. scaling the window (and bumping the subdivision limit) keeps the peak well inside the sampled range
def etaWindow(t, nu):
    width = 8.0 * np.sqrt(4 * nu * max(t, 1e-6))
    return max(width, 0.5) #still need enough range to see the cos(...) term when t is tiny

def coleHopfU(x, t, nu=main.NU):
    if t <= 0: #IC is exact, no need (and no way) to integrate through it
        return -np.sin(np.pi * x)

    span = etaWindow(t, nu)

    def phi(eta):
        return np.exp(-np.cos(np.pi * (x - eta)) / (2 * np.pi * nu) - (eta ** 2) / (4 * nu * t))

    def phiTimesGrad(eta):
        return np.sin(np.pi * (x - eta)) * phi(eta)

    numerator, _ = quad(phiTimesGrad, -span, span, limit=200)
    denominator, _ = quad(phi, -span, span, limit=200)

    return -numerator / denominator

def buildEvalGrid(nx=256, nt=100):
    xs = np.linspace(main.X_Min, main.X_Max, nx)
    ts = np.linspace(main.T_Min, main.T_Max, nt)
    return xs, ts

def relativeL2Error(uPred, uRef):
    uPred = np.array(uPred)
    uRef = np.array(uRef)
    return np.linalg.norm(uPred - uRef) / np.linalg.norm(uRef)

def evaluate(model, params, nx=256, nt=100):
    xs, ts = buildEvalGrid(nx, nt)

    uPred, uRef, residuals = [], [], []
    for t in ts:
        for x in xs:
            uPred.append(model.network(t, x, params))
            uRef.append(coleHopfU(x, t))
            residuals.append(model.pde_residual(t, x, params))

    l2Error = relativeL2Error(uPred, uRef)
    pdeError = np.mean(np.array(residuals) ** 2)
    return l2Error, pdeError

def logResult(resultsPath, config, l2Error, pdeError, checkpointPath):
    #tag every row with the model config + a timestamp so repeated runs append
    #instead of clobbering each other
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["timestamp", "checkpoint", "model", "n_qubits", "n_reuploads", "measured_qubits", "l2_error", "pde_residual_error"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            checkpointPath,
            config.get("model"),
            config.get("n_qubits"),
            config.get("n_reuploads"),
            config.get("measured_qubits"),
            l2Error,
            pdeError,
        ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pkl")
    parser.add_argument("--results", default="eval_results.csv")
    parser.add_argument("--model", default="main", help="module that defines network()/pde_residual() for this checkpoint, e.g. main or main_classical")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nt", type=int, default=100)
    args = parser.parse_args()

    modelModule = importlib.import_module(args.model)
    params, config = loadCheckpoint(args.checkpoint)
    l2Error, pdeError = evaluate(modelModule, params, nx=args.nx, nt=args.nt)

    print(f"config: {config}")
    print(f"relative L2 error vs Cole-Hopf: {l2Error:.6f}")
    print(f"PDE residual error (held-out grid): {pdeError:.6f}")

    logResult(args.results, config, l2Error, pdeError, args.checkpoint)
