import csv
import os

import main_classical
from sweep import trainAndEvalOnce, meanStd, DEFAULT_SEEDS, RESULTS_DIR, checkpointPathFor, DEFAULT_CHECKPOINT_DIR

#trains the classical control (main_classical.py) with the exact same seeds/epochs/eval-grid settings as the quantum sweep in sweep.py, so this lands in a row that's directly comparable - same columns, same grid, same number of seeds, just no quantum layer in the middle. model/modelTag/checkpointTag default to the original N_Qubits-wide control, but accept overrides so a capacity-matched control (main_classical.build_model(hidden_width=...)) can reuse this exact same training/eval path and land in the same CSV under its own tag
def runClassicalComparison(seeds=DEFAULT_SEEDS, epochs=100, nFBatch=20, evalNx=64, evalNt=50,
                            resultsPath=None, checkpointDir=DEFAULT_CHECKPOINT_DIR,
                            model=main_classical, modelTag="classical_control", checkpointTag="classical"):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "classical_comparison.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)
    os.makedirs(checkpointDir, exist_ok=True)

    runs = []
    for seed in seeds:
        checkpointPath = checkpointPathFor(None, None, checkpointDir, seed=seed, tag=checkpointTag)
        result = trainAndEvalOnce(model, seed, epochs=epochs, nFBatch=nFBatch,
                                   evalNx=evalNx, evalNt=evalNt, checkpointPath=checkpointPath)
        runs.append(result)
        print(f"seed={seed} l2={result['l2_error']:.4f} pdeErr={result['pde_residual_error']:.6f} "
              f"trainLoss={result['final_train_loss']:.6f} time={result['train_time_sec']:.1f}s", flush=True)

    l2Mean, l2Std = meanStd([r["l2_error"] for r in runs])
    pdeMean, pdeStd = meanStd([r["pde_residual_error"] for r in runs])
    lossMean, lossStd = meanStd([r["final_train_loss"] for r in runs])
    timeMean, timeStd = meanStd([r["train_time_sec"] for r in runs])

    print(f"{modelTag}: l2={l2Mean:.4f}+/-{l2Std:.4f} pdeErr={pdeMean:.6f}+/-{pdeStd:.6f}", flush=True)

    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["model", "n_seeds", "l2_error_mean", "l2_error_std",
                              "pde_residual_error_mean", "pde_residual_error_std",
                              "final_train_loss_mean", "final_train_loss_std",
                              "train_time_sec_mean", "train_time_sec_std",
                              "param_count", "peak_memory_mb"])
        writer.writerow([modelTag, len(seeds), l2Mean, l2Std, pdeMean, pdeStd,
                          lossMean, lossStd, timeMean, timeStd, runs[0]["param_count"], runs[-1]["peak_memory_mb"]])

    return runs

if __name__ == "__main__":
    runClassicalComparison()
