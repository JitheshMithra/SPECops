import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as onp
from pennylane import numpy as np

import main
import sweep
from checkpoint import loadCheckpoint

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

N_QUBITS = 5
N_REUPLOADS = 5
SEED = 0
N_DIRECTIONS = 3
ALPHAS = onp.linspace(-1.0, 1.0, 21)

#filter-normalized (Li et al. 2018) random direction: draw a random tensor matching each param tensor's shape, then rescale it so its norm matches that tensor's own norm - keeps the perturbation's relative scale sane across the very differently-sized W1/b1/W_q/W2/b2 tensors instead of one global random vector dominated by whichever tensor happens to be largest. uses plain numpy's RandomState (not pennylane's) since this is just generating perturbation vectors, not anything that needs autodiff tracking
def randomDirection(paramTuple, rngSeed):
    rng = onp.random.RandomState(rngSeed)
    direction = []
    for p in paramTuple:
        d = rng.randn(*p.shape)
        pNorm = onp.linalg.norm(p)
        dNorm = onp.linalg.norm(d)
        if dNorm > 0 and pNorm > 0:
            d = d * (pNorm / dNorm)
        direction.append(d)
    return direction

def perturb(paramTuple, direction, alpha):
    return tuple(p + alpha * d for p, d in zip(paramTuple, direction))

def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpointPath = sweep.checkpointPathFor(N_QUBITS, N_REUPLOADS, sweep.DEFAULT_CHECKPOINT_DIR, seed=SEED)
    params, config = loadCheckpoint(checkpointPath)
    print(f"loaded checkpoint {checkpointPath}: config={config}", flush=True)

    model = main.build_model(n_qubits=N_QUBITS, n_reuploads=N_REUPLOADS)
    #full collocation/data set, not a training mini-batch, so the slice reflects the actual converged loss surface rather than mini-batch sampling noise
    t_data, x_data, u_data, t_f, x_f = main.make_training_data()

    rows = []
    curves = {}
    for dirIdx in range(N_DIRECTIONS):
        direction = randomDirection(params, rngSeed=1000 + dirIdx)
        losses = []
        for alpha in ALPHAS:
            perturbedParams = perturb(params, direction, float(alpha))
            loss = float(model.loss_fn(perturbedParams, t_data, x_data, u_data, t_f, x_f))
            losses.append(loss)
            rows.append({"direction": dirIdx, "alpha": float(alpha), "loss": loss})
        curves[dirIdx] = losses
        centerLoss = losses[len(ALPHAS) // 2]
        print(f"direction {dirIdx}: loss range [{min(losses):.6f}, {max(losses):.6f}], loss at alpha=0 = {centerLoss:.6f}", flush=True)

    outCsv = os.path.join(RESULTS_DIR, "loss_landscape_slice.csv")
    with open(outCsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["direction", "alpha", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outCsv}", flush=True)

    plt.figure(figsize=(7, 5))
    for dirIdx, losses in curves.items():
        plt.plot(ALPHAS, losses, marker="o", markersize=3, label=f"direction {dirIdx}")
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("alpha (perturbation magnitude along random direction)")
    plt.ylabel("loss (full dataset)")
    plt.title(f"loss-landscape slice around q{N_QUBITS}_r{N_REUPLOADS} converged minimum")
    plt.legend()
    plt.tight_layout()
    outPng = os.path.join(RESULTS_DIR, "loss_landscape_slice.png")
    plt.savefig(outPng)
    plt.close()
    print(f"saved {outPng}", flush=True)
    return rows

if __name__ == "__main__":
    run()
