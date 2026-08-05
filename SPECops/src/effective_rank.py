import csv
import os

from pennylane import numpy as np

import main
import sweep
from checkpoint import loadCheckpoint
from activation_analysis import buildEvalPoints
from activation_diversity import collectActivationMatrices

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#effective rank via SVD, entropy-based definition (Roy & Vetterli 2007), applied here the same way "Learning to Maximize Quantum Neural Network Expressivity via Effective Rank" (arxiv 2506.15375) uses it as a QNN expressivity metric - this project is applying their metric to the post-quantum-layer output matrix, not inventing a new one
def effectiveRank(activationMatrix):
    singularValues = np.linalg.svd(activationMatrix, compute_uv=False)
    total = np.sum(singularValues)
    if total == 0:
        return 0.0
    normalized = singularValues / total
    nonzero = normalized[normalized > 0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(np.exp(entropy))

def run(configs, nx=20, nt=20, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "effective_rank.csv")
    evalPoints = buildEvalPoints(nx, nt)

    rows = []
    for label, nQubits, nReuploads, checkpointPath in configs:
        model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
        params, _ = loadCheckpoint(checkpointPath)
        _, postOut, _ = collectActivationMatrices(model, params, evalPoints)

        rank = effectiveRank(postOut)
        rows.append({"config": label, "n_qubits": nQubits, "n_reuploads": nReuploads, "effective_rank": rank})
        print(f"{label}: effective_rank={rank:.4f} (max possible = n_qubits = {nQubits})", flush=True)

    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "n_qubits", "n_reuploads", "effective_rank"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    #all 12 sweep configs (3 qubit counts x 4 reupload counts), not just the 4 corners - all seed=0 checkpoints already exist from sweep.py, no retraining needed
    configs = [
        (f"q{nQ}_r{nR}", nQ, nR, sweep.checkpointPathFor(nQ, nR, sweep.DEFAULT_CHECKPOINT_DIR, seed=0))
        for nQ in sweep.DEFAULT_QUBITS for nR in sweep.DEFAULT_REUPLOADS
    ]
    run(configs)
