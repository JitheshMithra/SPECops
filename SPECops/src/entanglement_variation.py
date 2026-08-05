import csv
import os
import time
from types import SimpleNamespace

import pennylane as qp
from pennylane import numpy as np

import main
import loop
import sweep
from checkpoint import loadCheckpoint
from eval import evaluate
from activation_analysis import buildEvalPoints
from activation_diversity import collectActivationMatrices, diversityMetrics
from effective_rank import effectiveRank

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entanglement_checkpoints")

N_QUBITS = 5
N_REUPLOADS = 5
SEED = 0
EPOCHS = 100 #matches sweep.py's original q5_r5 run so results are comparable to the existing ring checkpoint
MAX_ATTEMPTS = 2 #retry once on MemoryError, same pattern as run_heat_equation_remaining.sh - q5_r5 has MemoryError'd twice already on this project (extended-training run, heat-equation sweep)

#same encoding/training-block structure as main.py's build_model - only the entangling block differs per topology. copied rather than parameterized in main.py itself since main.py's quantum_circuit has the ring entangling gates hardcoded with no hook to swap them (same reasoning activation_diversity.py used for its mixed-observable variant)
def entanglingBlock(topology, n_qubits):
    if topology == "ring":
        for q in range(n_qubits):
            qp.CNOT(wires=[q, (q + 1) % n_qubits])
    elif topology == "linear":
        for q in range(n_qubits - 1): #chain, no wraparound CNOT back to qubit 0 - the only difference from ring
            qp.CNOT(wires=[q, q + 1])
    elif topology == "none":
        pass #no entangling gates at all
    else:
        raise ValueError(f"unknown entanglement topology: {topology}")

def build_model(topology, n_qubits=N_QUBITS, n_reuploads=N_REUPLOADS):
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def quantum_circuit(inputs, weights):
        for layer in range(n_reuploads):
            for q in range(n_qubits):
                qp.RY(inputs[q], wires=q)
            for q in range(n_qubits):
                qp.RY(weights[layer, q, 0], wires=q)
                qp.RZ(weights[layer, q, 1], wires=q)
            entanglingBlock(topology, n_qubits)
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    def init_params():
        W1 = np.random.randn(n_qubits, 2) * 0.1
        b1 = np.zeros(n_qubits)
        W_q = np.random.randn(n_reuploads, n_qubits, 2) * 0.1
        W2 = np.random.randn(1, n_qubits) * 0.1
        b2 = np.zeros(1)
        for p in (W1, b1, W_q, W2, b2):
            p.requires_grad = True
        return W1, b1, W_q, W2, b2

    PRE_LAYER_SCALE = np.pi

    def pre_layer(t, x, params):
        W1, b1 = params[0], params[1]
        inp = np.array([t, x])
        return np.tanh(W1 @ inp + b1) * PRE_LAYER_SCALE

    def network(t, x, params):
        W_q, W2, b2 = params[2], params[3], params[4]
        angles = pre_layer(t, x, params)
        q_out = quantum_circuit(angles, W_q)
        return (W2 @ np.array(q_out) + b2)[0]

    def pde_residual(t, x, params):
        u_of_t = lambda t_: network(t_, x, params)
        u_t = qp.grad(u_of_t, argnums=0)(t)
        u_of_x = lambda x_: network(t, x_, params)
        u_x_fn = qp.grad(u_of_x, argnums=0)
        u_x = u_x_fn(x)
        u_xx = qp.grad(u_x_fn, argnums=0)(x)
        u = network(t, x, params)
        return u_t + u * u_x - main.NU * u_xx

    def loss_fn(params, t_data, x_data, u_data, t_f, x_f):
        u_pred = np.array([network(t_data[i, 0], x_data[i, 0], params) for i in range(len(t_data))])
        mse_u = np.mean((u_pred - u_data[:, 0]) ** 2)
        f_pred = np.array([pde_residual(t_f[i, 0], x_f[i, 0], params) for i in range(len(t_f))])
        mse_f = np.mean(f_pred ** 2)
        return mse_u + mse_f

    def checkpoint_config():
        return {"model": "quantum_pinn_entanglement_variant", "n_qubits": n_qubits, "n_reuploads": n_reuploads,
                "measured_qubits": n_qubits, "entanglement": topology}

    return SimpleNamespace(
        n_qubits=n_qubits, n_reuploads=n_reuploads, entanglement=topology,
        quantum_circuit=quantum_circuit, init_params=init_params,
        network=network, pde_residual=pde_residual, loss_fn=loss_fn,
        checkpoint_config=checkpoint_config, pre_layer=pre_layer, pre_layer_scale=PRE_LAYER_SCALE,
    )

#fresh q5_r5 train under one topology, retrying once on MemoryError (q5_r5 has hit this twice already elsewhere in the project) - returns (model, params) or (None, None) if both attempts fail, so run() can log the failure and move on to the next topology instead of aborting
def trainWithRetry(topology, checkpointPath):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[entanglement={topology}] training attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
        try:
            np.random.seed(SEED)
            model = build_model(topology)
            params, _ = loop.train(epochs=EPOCHS, n_f_batch=20, model=model, checkpointPath=checkpointPath)
            return model, params
        except MemoryError:
            print(f"[entanglement={topology}] attempt {attempt}/{MAX_ATTEMPTS} failed with MemoryError", flush=True)
            if attempt == MAX_ATTEMPTS:
                print(f"[entanglement={topology}] FAILED after {MAX_ATTEMPTS} attempts, skipping", flush=True)
                return None, None

def diversityAndRank(model, params, evalPoints):
    _, postOut, _ = collectActivationMatrices(model, params, evalPoints)
    _, meanPairwiseCorrelation = diversityMetrics(postOut)
    rank = effectiveRank(postOut)
    return meanPairwiseCorrelation, rank

def run():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    evalPoints = buildEvalPoints(20, 20)
    rows = []

    #ring reuses the already-trained q5_r5 sweep checkpoint instead of retraining - main.py's default entangling block IS the ring topology, so this is the exact same architecture already on disk
    print("[entanglement=ring] reusing existing sweep checkpoint (main.py's default topology), not retraining", flush=True)
    ringCheckpoint = sweep.checkpointPathFor(N_QUBITS, N_REUPLOADS, sweep.DEFAULT_CHECKPOINT_DIR, seed=SEED)
    ringModel = main.build_model(n_qubits=N_QUBITS, n_reuploads=N_REUPLOADS)
    ringParams, _ = loadCheckpoint(ringCheckpoint)
    ringL2, ringPde = evaluate(ringModel, ringParams, nx=64, nt=50)
    ringCorr, ringRank = diversityAndRank(ringModel, ringParams, evalPoints)
    rows.append({"topology": "ring", "retrained": False, "l2_error": ringL2, "pde_residual_error": ringPde,
                 "mean_pairwise_correlation": ringCorr, "effective_rank": ringRank})
    print(f"[entanglement=ring] l2={ringL2:.6f} pdeErr={ringPde:.6f} meanPairwiseCorr={ringCorr:.6f} effRank={ringRank:.4f}", flush=True)

    for topology in ("linear", "none"):
        checkpointPath = os.path.join(CHECKPOINT_DIR, f"q{N_QUBITS}_r{N_REUPLOADS}_{topology}_s{SEED}.pkl")
        startTime = time.time()
        model, params = trainWithRetry(topology, checkpointPath)
        trainTime = time.time() - startTime

        if model is None:
            rows.append({"topology": topology, "retrained": True, "l2_error": None, "pde_residual_error": None,
                         "mean_pairwise_correlation": None, "effective_rank": None})
            continue

        l2, pde = evaluate(model, params, nx=64, nt=50)
        corr, rank = diversityAndRank(model, params, evalPoints)
        rows.append({"topology": topology, "retrained": True, "l2_error": l2, "pde_residual_error": pde,
                     "mean_pairwise_correlation": corr, "effective_rank": rank})
        print(f"[entanglement={topology}] l2={l2:.6f} pdeErr={pde:.6f} meanPairwiseCorr={corr:.6f} effRank={rank:.4f} trainTime={trainTime:.1f}s", flush=True)

    outPath = os.path.join(RESULTS_DIR, "entanglement_variation.csv")
    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "retrained", "l2_error", "pde_residual_error",
                                                "mean_pairwise_correlation", "effective_rank"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    run()
