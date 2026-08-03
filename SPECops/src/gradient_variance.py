import argparse
import csv
import os
import subprocess
import sys

import pennylane as qp
from pennylane import numpy as np

import main

DEFAULT_QUBITS = (3, 4, 5)
DEFAULT_REUPLOADS = (1, 2, 3, 5)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#one fixed small batch of collocation points, shared across every config and every random init, so the only thing that changes between measurements is the circuit architecture and the random init - not the data
def fixedCollocationBatch(batchSize=20):
    _, _, _, t_f, x_f = main.make_training_data()
    return t_f[:batchSize], x_f[:batchSize]

#pde-residual-only loss (no labelled data term) - this is the piece of the loss that actually routes through the quantum layer, which is what we care about for a barren plateau check; takes the params unpacked as separate positional args (not as one tuple) because qp.grad silently returns an empty gradient when the whole params tuple is passed as a single argument, it needs argnums pointed at one specific positional arg to work
def pdeLoss(model, W1, b1, W_q, W2, b2, tBatch, xBatch):
    params = (W1, b1, W_q, W2, b2)
    residuals = np.array([model.pde_residual(tBatch[i, 0], xBatch[i, 0], params)
                           for i in range(len(tBatch))])
    return np.mean(residuals ** 2)

#samples nSamples random inits, computes the gradient of the PDE-residual loss w.r.t. the quantum layer's weights (W_q, the 3rd positional arg) at each one, and returns the per-component gradient variance across inits, averaged into one scalar - a value collapsing toward zero as n_qubits grows is the barren-plateau signature. batchSize defaults small (5, not 20) because this loss nests 3 levels of autodiff (see runSingleConfigSubprocess note below) - at batchSize=20 the per-sample computation graph is big enough that looping it 100 times inside one process runs out of memory on deeper-circuit configs, even in isolation
def gradientVariance(nQubits, nReuploads, nSamples=50, batchSize=5):
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
    tBatch, xBatch = fixedCollocationBatch(batchSize)

    costFn = lambda W1, b1, W_q, W2, b2: pdeLoss(model, W1, b1, W_q, W2, b2, tBatch, xBatch)
    gradFn = qp.grad(costFn, argnums=2) #W_q is the 3rd positional arg

    wqGrads = []
    for _ in range(nSamples):
        params = model.init_params()
        wqGrads.append(gradFn(*params))

    wqGrads = np.stack(wqGrads) #shape (nSamples, n_reuploads, n_qubits, 2)
    perComponentVariance = np.var(wqGrads, axis=0)
    return float(np.mean(perComponentVariance))

#this loss nests THREE levels of autodiff (pde_residual already does d/dx and d2/dx2 through the circuit internally, then this differentiates that whole thing again w.r.t. W_q) - at 100 samples that graph accumulation blew out memory partway through a real run (MemoryError inside autograd's backward pass on the 2nd config), so running each config in its own subprocess isolates the memory instead of trying to chase down autograd's internal retention
def runSingleConfigSubprocess(nQubits, nReuploads, nSamples, batchSize):
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--single",
         "--n-qubits", str(nQubits), "--n-reuploads", str(nReuploads),
         "--n-samples", str(nSamples), "--batch-size", str(batchSize)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"n_qubits={nQubits} n_reuploads={nReuploads} subprocess failed:\n{result.stderr[-2000:]}")
    return float(result.stdout.strip().splitlines()[-1])

def runGradientVarianceSweep(nQubitsList=DEFAULT_QUBITS, nReuploadsList=DEFAULT_REUPLOADS,
                              nSamples=50, batchSize=5, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "gradient_variance_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_reuploads", "mean_grad_variance"])

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                try:
                    variance = runSingleConfigSubprocess(nQubits, nReuploads, nSamples, batchSize)
                except RuntimeError as e:
                    print(f"n_qubits={nQubits} n_reuploads={nReuploads} FAILED: {e}", flush=True)
                    writer.writerow([nQubits, nReuploads, "FAILED"])
                    f.flush()
                    continue

                print(f"n_qubits={nQubits} n_reuploads={nReuploads} mean_grad_variance={variance:.3e}", flush=True)
                writer.writerow([nQubits, nReuploads, variance])
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="internal: compute one config and print its variance, used by the subprocess-isolated sweep driver")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--n-reuploads", type=int)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    if args.single:
        print(gradientVariance(args.n_qubits, args.n_reuploads, nSamples=args.n_samples, batchSize=args.batch_size))
    else:
        runGradientVarianceSweep()
