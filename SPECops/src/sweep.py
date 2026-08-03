import csv
import os

import main
import loop
from eval import evaluate

#loops over (n_qubits, n_reuploads) combos, trains each one through loop.py's training loop, scores it with eval.py's metrics, and appends one row per config to a results CSV so a run that dies partway through doesn't lose earlier rows
def runSweep(nQubitsList=(3, 4, 5), nReuploadsList=(1, 2, 3, 5), epochs=100, nFBatch=20,
             evalNx=256, evalNt=100, resultsPath="sweep_results.csv"):
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_reuploads", "l2_error", "pde_residual_error", "final_train_loss"])

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                print(f"=== training n_qubits={nQubits} n_reuploads={nReuploads} ===")
                model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
                params, history = loop.train(epochs=epochs, n_f_batch=nFBatch, model=model)
                l2Error, pdeError = evaluate(model, params, nx=evalNx, nt=evalNt)

                print(f"n_qubits={nQubits} n_reuploads={nReuploads} l2={l2Error:.4f} pdeErr={pdeError:.6f} trainLoss={history[-1]:.6f}")
                writer.writerow([nQubits, nReuploads, l2Error, pdeError, history[-1]])
                f.flush() #keep partial progress on disk in case a later config crashes or gets killed

if __name__ == "__main__":
    runSweep()
