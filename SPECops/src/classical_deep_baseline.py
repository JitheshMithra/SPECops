import argparse
import os
import time
from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np

from main import make_training_data, NU, N_U
from eval import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#a proper deep tanh MLP PINN (no quantum layer, no shape-matching to main_classical.py's single-hidden-layer control) - built to check whether classical training can reach normal Raissi-et-al-level accuracy on this exact Burgers setup at all, given Adam + depth + enough epochs, or whether ~0.9 L2 is a training-setup ceiling rather than an architecture one
def buildDeepModel(hiddenLayers=8, hiddenWidth=20):
    layerSizes = [2] + [hiddenWidth] * hiddenLayers + [1]

    def init_params():
        params = []
        for inSize, outSize in zip(layerSizes[:-1], layerSizes[1:]):
            #glorot-ish scale (1/sqrt(fan_in)) instead of the flat *0.1 used elsewhere in this repo - a flat 0.1 std saturates every tanh once the net gets this deep, which would defeat the point of testing whether depth helps
            W = np.random.randn(outSize, inSize) * np.sqrt(1.0 / inSize)
            b = np.zeros(outSize)
            W.requires_grad = True
            b.requires_grad = True
            params.append(W)
            params.append(b)
        return tuple(params)

    #batched forward pass: tBatch/xBatch are 1-D arrays of N points, not single scalars - this is what makes the residual computation below cheap, one matmul per layer over the whole batch instead of a python loop calling network() once per point
    def networkBatched(tBatch, xBatch, params):
        h = np.stack([tBatch, xBatch], axis=1) #shape (N, 2)
        nLayers = len(layerSizes) - 1
        for i in range(nLayers):
            W, b = params[2 * i], params[2 * i + 1]
            h = h @ W.T + b
            if i < nLayers - 1: #no activation on the final linear output layer
                h = np.tanh(h)
        return h[:, 0] #shape (N,)

    def network(t, x, params): #scalar convenience wrapper so this still matches the interface eval.py expects
        return float(networkBatched(np.array([t]), np.array([x]), params)[0])

    #sum-trick batched derivatives: since each output only depends on its own row's inputs (no cross-sample mixing in an MLP applied row-wise), grad of sum(outputs) w.r.t. the input vector gives the elementwise per-sample derivative in one autodiff call, instead of N separate calls
    def pdeResidualBatched(tBatch, xBatch, params):
        uOfTSum = lambda t_: np.sum(networkBatched(t_, xBatch, params))
        uT = qp.grad(uOfTSum, argnums=0)(tBatch)

        uOfXSum = lambda x_: np.sum(networkBatched(tBatch, x_, params))
        uXFn = qp.grad(uOfXSum, argnums=0)
        uX = uXFn(xBatch)
        uXXSum = lambda x_: np.sum(uXFn(x_))
        uXX = qp.grad(uXXSum, argnums=0)(xBatch)

        u = networkBatched(tBatch, xBatch, params)
        return uT + u * uX - NU * uXX

    def pde_residual(t, x, params): #scalar convenience wrapper, same reasoning as network() above
        return float(pdeResidualBatched(np.array([t]), np.array([x]), params)[0])

    def checkpoint_config():
        return {"model": "classical_deep_baseline", "n_qubits": None, "n_reuploads": None,
                "measured_qubits": None, "hidden_layers": hiddenLayers, "hidden_width": hiddenWidth}

    return SimpleNamespace(hidden_layers=hiddenLayers, hidden_width=hiddenWidth,
                            init_params=init_params, network=network, pde_residual=pde_residual,
                            networkBatched=networkBatched, pdeResidualBatched=pdeResidualBatched,
                            checkpoint_config=checkpoint_config)

#splits the combined IC+BC loss main.make_training_data() returns back into its two pieces (IC is the first N_U//2 rows, BC is the rest) purely for reporting - training still optimizes IC+BC+residual jointly like every other model in this repo, this just shows whether one term dominates or is under-weighted, which the combined MSE_u number alone would hide
def lossComponents(model, params, tData, xData, uData, tF, xF, nIc):
    uPred = model.networkBatched(tData, xData, params)
    icLoss = float(np.mean((uPred[:nIc] - uData[:nIc]) ** 2))
    bcLoss = float(np.mean((uPred[nIc:] - uData[nIc:]) ** 2))
    residualLoss = float(np.mean(model.pdeResidualBatched(tF, xF, params) ** 2))
    return icLoss, bcLoss, residualLoss

def train(model, epochs=5000, lr=1e-3, nFBatch=20, reportEvery=500):
    params = model.init_params()
    opt = qp.AdamOptimizer(stepsize=lr)

    tData, xData, uData, tF, xF = make_training_data()
    tData, xData, uData = tData[:, 0], xData[:, 0], uData[:, 0]
    tF, xF = tF[:, 0], xF[:, 0]
    nIc = N_U // 2

    startTime = time.time()
    for epoch in range(1, epochs + 1):
        idx = np.random.choice(len(tF), size=nFBatch, replace=False)
        tBatch, xBatch = tF[idx], xF[idx]

        def cost(*p):
            uPred = model.networkBatched(tData, xData, p)
            mseU = np.mean((uPred - uData) ** 2)
            mseF = np.mean(model.pdeResidualBatched(tBatch, xBatch, p) ** 2)
            return mseU + mseF

        params = opt.step(cost, *params)

        if epoch % reportEvery == 0 or epoch == 1:
            icLoss, bcLoss, residualLoss = lossComponents(model, params, tData, xData, uData, tF, xF, nIc)
            elapsed = time.time() - startTime
            print(f"epoch {epoch:5d}  ic={icLoss:.6f}  bc={bcLoss:.6f}  residual={residualLoss:.6f}  "
                  f"total={icLoss + bcLoss + residualLoss:.6f}  elapsed={elapsed:.1f}s", flush=True)

    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-layers", type=int, default=8)
    parser.add_argument("--hidden-width", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-f-batch", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=500)
    args = parser.parse_args()

    model = buildDeepModel(hiddenLayers=args.hidden_layers, hiddenWidth=args.hidden_width)
    print(f"deep classical baseline: {args.hidden_layers} hidden layers x {args.hidden_width} neurons, Adam lr={args.lr}, {args.epochs} epochs", flush=True)

    startTime = time.time()
    params = train(model, epochs=args.epochs, lr=args.lr, nFBatch=args.n_f_batch, reportEvery=args.report_every)
    trainTime = time.time() - startTime
    print(f"training finished in {trainTime:.1f}s", flush=True)

    l2Error, pdeError = evaluate(model, params, nx=64, nt=50)
    print(f"FINAL relative L2 error vs Cole-Hopf: {l2Error:.6f}", flush=True)
    print(f"FINAL PDE residual error: {pdeError:.6f}", flush=True)
