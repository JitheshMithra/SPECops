import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pennylane import numpy as np

import main
import sweep
from checkpoint import loadCheckpoint
from eval import buildEvalGrid, coleHopfU

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#checks the hypothesis behind the PDE-residual/L2 anti-correlation: does a low-expressivity config collapse toward a near-flat solution that trivially satisfies the PDE residual while being far from the true (sharply-varying, shocked) solution? predicts the flat solution's total variation and gradient magnitude, computed from the network's own prediction grid, not from any autodiff - a coarse finite-difference on the evaluated surface is enough to answer "is this flatter than the reference"
def predictGrid(model, params, nx=128, nt=60):
    xs, ts = buildEvalGrid(nx, nt)
    uPred = np.zeros((len(ts), len(xs)))
    uRef = np.zeros((len(ts), len(xs)))
    for i, t in enumerate(ts):
        for j, x in enumerate(xs):
            uPred[i, j] = float(model.network(t, x, params))
            uRef[i, j] = float(coleHopfU(x, t))
    return xs, ts, uPred, uRef

#total variation in x, averaged over t - sum of |du/dx| along each time slice via finite differences on the already-evaluated grid, then averaged across all time slices, so this is a single scalar summarizing "how much the surface wiggles in space" per config
def totalVariation(u):
    diffs = np.abs(np.diff(u, axis=1)) #along x
    return float(np.mean(np.sum(diffs, axis=1)))

def meanAbsGradient(u, xs):
    dx = float(xs[1] - xs[0])
    grad = np.diff(u, axis=1) / dx
    return float(np.mean(np.abs(grad)))

def run():
    configs = [
        ("q3_r1", 3, 1, sweep.checkpointPathFor(3, 1, sweep.DEFAULT_CHECKPOINT_DIR, seed=0)),
        ("q5_r5", 5, 5, sweep.checkpointPathFor(5, 5, sweep.DEFAULT_CHECKPOINT_DIR, seed=0)),
    ]

    results = []
    grids = {}
    xsShared, tsShared, uRefShared = None, None, None

    for label, nQubits, nReuploads, checkpointPath in configs:
        model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
        params, config = loadCheckpoint(checkpointPath)
        xs, ts, uPred, uRef = predictGrid(model, params)
        xsShared, tsShared, uRefShared = xs, ts, uRef

        tv = totalVariation(uPred)
        gradMag = meanAbsGradient(uPred, xs)
        ampRange = float(np.max(uPred) - np.min(uPred))

        grids[label] = uPred
        results.append({"config": label, "total_variation": tv, "mean_abs_gradient": gradMag,
                         "amplitude_range": ampRange})
        print(f"{label}: total_variation={tv:.4f} mean_abs_gradient={gradMag:.4f} amplitude_range={ampRange:.4f}", flush=True)

    refTv = totalVariation(uRefShared)
    refGrad = meanAbsGradient(uRefShared, xsShared)
    refAmp = float(np.max(uRefShared) - np.min(uRefShared))
    results.append({"config": "cole_hopf_reference", "total_variation": refTv,
                     "mean_abs_gradient": refGrad, "amplitude_range": refAmp})
    print(f"reference: total_variation={refTv:.4f} mean_abs_gradient={refGrad:.4f} amplitude_range={refAmp:.4f}", flush=True)

    outCsv = os.path.join(RESULTS_DIR, "flatness_check.csv")
    with open(outCsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "total_variation", "mean_abs_gradient", "amplitude_range"])
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {outCsv}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, u) in zip(axes, [("q3_r1 (worst, lowest residual)", grids["q3_r1"]),
                                       ("q5_r5 (best, highest residual)", grids["q5_r5"]),
                                       ("Cole-Hopf reference", uRefShared)]):
        im = ax.pcolormesh(xsShared, tsShared, u, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("x")
        ax.set_title(label)
    axes[0].set_ylabel("t")
    fig.colorbar(im, ax=axes, label="u(x,t)")
    plotPath = os.path.join(RESULTS_DIR, "flatness_check.png")
    plt.savefig(plotPath)
    plt.close()
    print(f"saved {plotPath}", flush=True)

    return results

if __name__ == "__main__":
    run()
