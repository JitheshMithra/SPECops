import csv
import os

import measurement_operator_variation as movar
from checkpoint import loadCheckpoint
from activation_analysis import buildEvalPoints
from activation_diversity import collectActivationMatrices, diversityMetrics
from effective_rank import effectiveRank

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#same Section 6 diversity metrics applied to the Tier 5B probs checkpoint - flagged explicitly (in both the printed note and the saved notes column) that this output is 2^n_qubits-dimensional, not n_qubits-dimensional, so it isn't shape-comparable to the ring/linear/none PauliZ configs the same metrics were computed for elsewhere
def run(nx=20, nt=20, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "measurement_operator_redundancy.csv")
    checkpointPath = os.path.join(movar.CHECKPOINT_DIR, f"q{movar.N_QUBITS}_r{movar.N_REUPLOADS}_probs_s{movar.SEED}.pkl")
    if not os.path.exists(checkpointPath):
        print(f"no probs checkpoint at {checkpointPath} - run measurement_operator_variation.py first", flush=True)
        return []

    model = movar.build_model()
    params, _ = loadCheckpoint(checkpointPath)
    evalPoints = buildEvalPoints(nx, nt)
    _, postOut, _ = collectActivationMatrices(model, params, evalPoints)
    _, meanPairwiseCorrelation = diversityMetrics(postOut)
    rank = effectiveRank(postOut)

    note = (f"probs-based output, dimensionality {model.output_dim} (2^{movar.N_QUBITS}) vs {movar.N_QUBITS} for "
            f"the pauliz configs - not directly shape-comparable to the ring/linear/none PauliZ results")
    print(f"measurement=probs: mean_pairwise_correlation={meanPairwiseCorrelation:.6f} effective_rank={rank:.4f}", flush=True)
    print(f"NOTE: {note}", flush=True)

    rows = [{"measurement": "probs", "output_dim": model.output_dim,
             "mean_pairwise_correlation": meanPairwiseCorrelation, "effective_rank": rank, "notes": note}]
    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["measurement", "output_dim", "mean_pairwise_correlation", "effective_rank", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    run()
