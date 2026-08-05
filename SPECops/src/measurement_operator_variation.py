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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurement_checkpoints")

N_QUBITS = 5
N_REUPLOADS = 5
SEED = 0
EPOCHS = 100 #matches sweep.py's original q5_r5 run so results are comparable to the existing pauliz checkpoint
MAX_ATTEMPTS = 2 #retry once on MemoryError, same pattern as run_heat_equation_remaining.sh - q5_r5 has MemoryError'd twice already on this project

#same ring-entangled gate sequence as main.py's build_model - only the measurement changes: qml.probs() over all wires instead of one PauliZ expval per wire. that turns the quantum layer's output from n_qubits numbers into 2^n_qubits numbers (a full probability distribution), so the post-layer's input size (W2 below) is sized off outputDim, not n_qubits - called out explicitly in run()'s printed output, not silently reshaped/truncated
def build_model(n_qubits=N_QUBITS, n_reuploads=N_REUPLOADS):
    dev = qp.device("default.qubit", wires=n_qubits)
    outputDim = 2 ** n_qubits

    @qp.qnode(dev)
    def quantum_circuit(inputs, weights):
        for layer in range(n_reuploads):
            for q in range(n_qubits):
                qp.RY(inputs[q], wires=q)
            for q in range(n_qubits):
                qp.RY(weights[layer, q, 0], wires=q)
                qp.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])
        return qp.probs(wires=range(n_qubits))

    def init_params():
        W1 = np.random.randn(n_qubits, 2) * 0.1
        b1 = np.zeros(n_qubits)
        W_q = np.random.randn(n_reuploads, n_qubits, 2) * 0.1
        W2 = np.random.randn(1, outputDim) * 0.1 #one weight per basis-state probability - outputDim = 2^n_qubits, not n_qubits
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
        return {"model": "quantum_pinn_probs_variant", "n_qubits": n_qubits, "n_reuploads": n_reuploads,
                "measured_qubits": n_qubits, "measurement": "probs", "output_dim": outputDim}

    return SimpleNamespace(
        n_qubits=n_qubits, n_reuploads=n_reuploads, output_dim=outputDim, measurement="probs",
        quantum_circuit=quantum_circuit, init_params=init_params,
        network=network, pde_residual=pde_residual, loss_fn=loss_fn,
        checkpoint_config=checkpoint_config, pre_layer=pre_layer, pre_layer_scale=PRE_LAYER_SCALE,
    )

#sanity check on circuit wiring before trusting any training result - probs() should sum to 1 per sample. uses a throwaway seed (999) for the dummy params draw so this check doesn't disturb the RNG stream that the real training run below reseeds to SEED right before it
def checkProbsSumToOne(nSamples=5):
    np.random.seed(999)
    model = build_model()
    dummyParams = model.init_params()
    xs = np.linspace(main.X_Min, main.X_Max, nSamples)
    ts = np.linspace(main.T_Min, main.T_Max, nSamples)
    for t, x in zip(ts, xs):
        angles = model.pre_layer(t, x, dummyParams)
        probs = model.quantum_circuit(angles, dummyParams[2])
        total = float(np.sum(probs))
        if not np.isclose(total, 1.0, atol=1e-6):
            raise AssertionError(f"probs output sums to {total} at (t={t:.4f}, x={x:.4f}), expected 1.0 - circuit wiring is broken")
    print(f"sanity check passed: probs() output sums to 1.0 across {nSamples} sample points", flush=True)

def trainWithRetry(checkpointPath):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[measurement=probs] training attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
        try:
            np.random.seed(SEED)
            model = build_model()
            params, _ = loop.train(epochs=EPOCHS, n_f_batch=20, model=model, checkpointPath=checkpointPath)
            return model, params
        except MemoryError:
            print(f"[measurement=probs] attempt {attempt}/{MAX_ATTEMPTS} failed with MemoryError", flush=True)
            if attempt == MAX_ATTEMPTS:
                print(f"[measurement=probs] FAILED after {MAX_ATTEMPTS} attempts, skipping", flush=True)
                return None, None

def run():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    checkProbsSumToOne()

    #pauliz baseline reuses the existing sweep checkpoint instead of retraining
    print("[measurement=pauliz] reusing existing sweep checkpoint, not retraining", flush=True)
    pauliZCheckpoint = sweep.checkpointPathFor(N_QUBITS, N_REUPLOADS, sweep.DEFAULT_CHECKPOINT_DIR, seed=SEED)
    pauliZModel = main.build_model(n_qubits=N_QUBITS, n_reuploads=N_REUPLOADS)
    pauliZParams, _ = loadCheckpoint(pauliZCheckpoint)
    pauliZL2, pauliZPde = evaluate(pauliZModel, pauliZParams, nx=64, nt=50)
    pauliZCount = sweep.paramCount(pauliZParams)
    rows = [{"measurement": "pauliz", "output_dim": N_QUBITS, "param_count": pauliZCount,
             "l2_error": pauliZL2, "pde_residual_error": pauliZPde, "notes": "baseline, reused from sweep checkpoint"}]
    print(f"[measurement=pauliz] outputDim={N_QUBITS} paramCount={pauliZCount} l2={pauliZL2:.6f} pdeErr={pauliZPde:.6f}", flush=True)

    checkpointPath = os.path.join(CHECKPOINT_DIR, f"q{N_QUBITS}_r{N_REUPLOADS}_probs_s{SEED}.pkl")
    startTime = time.time()
    model, params = trainWithRetry(checkpointPath)
    trainTime = time.time() - startTime

    if model is None:
        rows.append({"measurement": "probs", "output_dim": 2 ** N_QUBITS, "param_count": None,
                     "l2_error": None, "pde_residual_error": None, "notes": "training failed after retries"})
    else:
        l2, pde = evaluate(model, params, nx=64, nt=50)
        probsCount = sweep.paramCount(params)
        delta = probsCount - pauliZCount
        note = (f"post-layer input size = {model.output_dim} (2^{N_QUBITS}) vs {N_QUBITS} for pauliz baseline - "
                f"param count {pauliZCount} -> {probsCount} (delta={delta:+d}); breaks the parameter-matched "
                f"comparison story from Section 9, which assumed post-layer input size = n_qubits")
        print(f"[measurement=probs] outputDim={model.output_dim} paramCount={probsCount} l2={l2:.6f} pdeErr={pde:.6f} trainTime={trainTime:.1f}s", flush=True)
        print(f"[measurement=probs] WARNING: {note}", flush=True)
        rows.append({"measurement": "probs", "output_dim": model.output_dim, "param_count": probsCount,
                     "l2_error": l2, "pde_residual_error": pde, "notes": note})

    outPath = os.path.join(RESULTS_DIR, "measurement_operator_variation.csv")
    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["measurement", "output_dim", "param_count", "l2_error", "pde_residual_error", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

if __name__ == "__main__":
    run()
