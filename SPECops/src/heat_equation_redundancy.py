import csv
import os

import heat_equation as heq
from checkpoint import loadCheckpoint
from activation_analysis import buildEvalPoints
from activation_diversity import collectActivationMatrices, diversityMetrics
from effective_rank import effectiveRank

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heat_equation_checkpoints")

QUBITS = (3, 4, 5)
REUPLOADS = (1, 2, 3, 5)
SEEDS = (0, 1)

#same post-quantum-layer diversity metrics as Section 6 (mean_pairwise_correlation, effective_rank), applied to the heat-equation cross-validation checkpoints instead of the Burgers sweep - heat_equation.build_model() reuses main.build_model()'s architecture wholesale (same pre_layer/quantum_circuit/params layout), so collectActivationMatrices works against these checkpoints unmodified
def run(nx=20, nt=20, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "heat_equation_redundancy.csv")
    evalPoints = buildEvalPoints(nx, nt)

    rows = []
    for seed in SEEDS:
        for nQubits in QUBITS:
            for nReuploads in REUPLOADS:
                label = f"q{nQubits}_r{nReuploads}"
                checkpointPath = os.path.join(CHECKPOINT_DIR, f"{label}_s{seed}.pkl")
                if not os.path.exists(checkpointPath):
                    print(f"{label} seed={seed}: no checkpoint at {checkpointPath}, skipping", flush=True)
                    continue

                model = heq.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
                params, _ = loadCheckpoint(checkpointPath)
                _, postOut, _ = collectActivationMatrices(model, params, evalPoints)
                _, meanPairwiseCorrelation = diversityMetrics(postOut)
                rank = effectiveRank(postOut)

                rows.append({"config": label, "seed": seed, "n_qubits": nQubits, "n_reuploads": nReuploads,
                             "mean_pairwise_correlation": meanPairwiseCorrelation, "effective_rank": rank})
                print(f"{label} seed={seed}: mean_pairwise_correlation={meanPairwiseCorrelation:.6f} effective_rank={rank:.4f}", flush=True)

    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "seed", "n_qubits", "n_reuploads", "mean_pairwise_correlation", "effective_rank"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    run()
