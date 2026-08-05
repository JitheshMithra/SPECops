import csv
import os

from pennylane import numpy as np

import main
import sweep
from checkpoint import loadCheckpoint
from activation_analysis import buildEvalPoints
from activation_diversity import collectActivationMatrices, diversityMetrics
from effective_rank import effectiveRank

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#same post-quantum-layer diversity metrics as Section 6 (mean_pairwise_correlation via activation_diversity.diversityMetrics, effective_rank via effective_rank.effectiveRank), computed at training-epoch snapshots instead of just the final checkpoint, to see whether the redundancy signature is present from the start or emerges/resolves during training - q3_r1 and q5_r5 only (the two configs already covered elsewhere in the report), not all four corners
def snapshotMetrics(checkpointPath, nQubits, nReuploads, evalPoints):
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
    params, _ = loadCheckpoint(checkpointPath)
    _, postOut, _ = collectActivationMatrices(model, params, evalPoints)

    _, meanPairwiseCorrelation = diversityMetrics(postOut)
    rank = effectiveRank(postOut)
    return meanPairwiseCorrelation, rank

def run(configs, epochSnapshots=(100, 200, 300, 400, 500), nx=20, nt=20, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "redundancy_over_epochs.csv")
    evalPoints = buildEvalPoints(nx, nt)

    rows = []
    for label, nQubits, nReuploads in configs:
        for epoch in epochSnapshots:
            suffix = "" if epoch == 100 else f"_e{epoch}" #100-epoch checkpoints come straight from sweep.py (no suffix), later snapshots are extend_training.py's --out-checkpoint naming
            checkpointPath = os.path.join(sweep.DEFAULT_CHECKPOINT_DIR, f"{label}_s0{suffix}.pkl")

            meanPairwiseCorrelation, rank = snapshotMetrics(checkpointPath, nQubits, nReuploads, evalPoints)
            rows.append({"config": label, "epoch": epoch,
                         "mean_pairwise_correlation": meanPairwiseCorrelation, "effective_rank": rank})
            print(f"{label} epoch={epoch}: mean_pairwise_correlation={meanPairwiseCorrelation:.6f} effective_rank={rank:.4f}", flush=True)

    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "epoch", "mean_pairwise_correlation", "effective_rank"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    configs = [("q3_r1", 3, 1), ("q5_r5", 5, 5)]
    run(configs)
