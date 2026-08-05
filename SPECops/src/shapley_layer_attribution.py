import csv
import itertools
import os

import pennylane as qp
from pennylane import numpy as np

import sweep
from checkpoint import loadCheckpoint
from eval import buildEvalGrid, coleHopfU, relativeL2Error

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#Shapley-value attribution of the n_reuploads re-upload layers' contribution to L2 accuracy, applying the ablation approach of "Explaining Quantum Circuits with Shapley Values" (arXiv 2301.09138) to this project's checkpoint rather than proposing a new method - players are whole re-upload layers (not individual gates), value function is relative-L2 error improvement from activating a given subset of layers vs the empty (all-bypassed) circuit
PRE_LAYER_SCALE = np.pi #matches main.py's build_model

#same gate sequence as main.py's quantum_circuit (copied deliberately, same reasoning as activation_diversity.py's buildMixedObservableCircuit - build_model()'s qnode has no hook to skip a layer), except a layer not in activeLayers is skipped entirely: no encoding, no rotation, no entangling for that layer, so the state is passed through that block completely unmodified rather than zeroed or reset
def buildAblatedCircuit(n_qubits, n_reuploads, activeLayers):
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit(inputs, weights):
        for layer in range(n_reuploads):
            if layer not in activeLayers:
                continue
            for q in range(n_qubits):
                qp.RY(inputs[q], wires=q)
            for q in range(n_qubits):
                qp.RY(weights[layer, q, 0], wires=q)
                qp.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qp.expval(qp.PauliZ(i)) for i in range(n_qubits)]

    return circuit

def preLayer(t, x, W1, b1):
    inp = np.array([t, x])
    return np.tanh(W1 @ inp + b1) * PRE_LAYER_SCALE

#L2 error of the network with only `activeLayers` active, evaluated on the held-out (t,x) grid already used to score this checkpoint elsewhere in the project (extend_training.py's default 64x50 eval grid - matching it lets the full-circuit subset reproduce that checkpoint's already-recorded L2 exactly, which is the primary sanity check that the ablation mechanism itself is correct)
def l2ErrorForSubset(activeLayers, params, xs, ts, n_qubits, n_reuploads):
    W1, b1, W_q, W2, b2 = params
    circuit = buildAblatedCircuit(n_qubits, n_reuploads, activeLayers)

    uPred, uRef = [], []
    for t in ts:
        for x in xs:
            angles = preLayer(t, x, W1, b1)
            qOut = circuit(angles, W_q)
            u = (W2 @ np.array(qOut) + b2)[0]
            uPred.append(float(u))
            uRef.append(coleHopfU(x, t))

    return relativeL2Error(uPred, uRef)

def allSubsets(n):
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            yield frozenset(combo)

#exact Shapley value via full enumeration over all 5! orderings (32 subsets already cover every v(S) needed - enumerating permutations directly rather than the closed-form subset-weight formula, so this stays a literal average marginal contribution over orderings, easy to audit against the arXiv 2301.09138 definition
def exactShapley(nPlayers, value):
    contributions = {i: [] for i in range(nPlayers)}
    for perm in itertools.permutations(range(nPlayers)):
        seen = frozenset()
        for player in perm:
            before = value(seen)
            after = value(seen | {player})
            contributions[player].append(after - before)
            seen = seen | {player}
    return {i: float(np.mean(vals)) for i, vals in contributions.items()}

def run():
    #q5_r5_s0_e500.pkl: seed=0, plain-GD run extended to 500 epochs (the "GD run, L2=0.594" checkpoint), not the Adam run
    checkpointPath = os.path.join(sweep.DEFAULT_CHECKPOINT_DIR, "q5_r5_s0_e500.pkl")
    params, config = loadCheckpoint(checkpointPath)
    nQubits, nReuploads = config["n_qubits"], config["n_reuploads"]
    print(f"loaded {checkpointPath}: config={config}", flush=True)

    xs, ts = buildEvalGrid(nx=64, nt=50)

    l2BySubset = {}
    for subset in allSubsets(nReuploads):
        l2 = l2ErrorForSubset(subset, params, xs, ts, nQubits, nReuploads)
        l2BySubset[subset] = l2
        label = "{" + ",".join(str(i) for i in sorted(subset)) + "}"
        print(f"S={label:16s} l2_error={l2:.6f}", flush=True)

    emptyL2 = l2BySubset[frozenset()]
    fullL2 = l2BySubset[frozenset(range(nReuploads))]
    print(f"empty-circuit L2 = {emptyL2:.6f}, full-circuit L2 = {fullL2:.6f}, gap = {emptyL2 - fullL2:.6f}", flush=True)

    #value(S) = L2-error improvement of subset S over the empty (all-bypassed) circuit - the "relative L2-error improvement from including a subset vs excluding it" the design asks for; value(empty)=0 by construction, value(full)=the full-vs-empty gap above
    def value(subset):
        return emptyL2 - l2BySubset[subset]

    shapleyValues = exactShapley(nReuploads, value)

    subsetCsvPath = os.path.join(RESULTS_DIR, "shapley_subset_l2.csv")
    with open(subsetCsvPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["active_layers", "l2_error"])
        for subset, l2 in l2BySubset.items():
            writer.writerow([";".join(str(i) for i in sorted(subset)), l2])
    print(f"saved {subsetCsvPath}", flush=True)

    outCsvPath = os.path.join(RESULTS_DIR, "shapley_layer_attribution.csv")
    with open(outCsvPath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_index", "shapley_value"])
        for i in range(nReuploads):
            writer.writerow([i, shapleyValues[i]])
    print(f"saved {outCsvPath}", flush=True)

    shapleySum = sum(shapleyValues.values())
    print(f"sum of shapley values = {shapleySum:.6f} (efficiency check: should equal full-vs-empty gap = {emptyL2 - fullL2:.6f})", flush=True)
    for i in range(nReuploads):
        print(f"layer {i}: shapley_value={shapleyValues[i]:.6f}", flush=True)

    return l2BySubset, shapleyValues

if __name__ == "__main__":
    run()
