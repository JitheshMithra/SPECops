import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pennylane import numpy as np

import main
import main_classical
from checkpoint import loadCheckpoint
from eval import buildEvalGrid

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#on scope: this repo's network only has two real layers around the quantum block - pre-layer (encoding) -> quantum_circuit -> post-layer (readout) - there's no multi-layer classical head to match a "layers 2 and 3" figure from outside material this script doesn't have access to, so instead this compares the layer that actually sits right after where the quantum circuit would go: the quantum_circuit's own output for the QAPINN, vs a straight pass-through of the pre-layer for the classical control (since main_classical.py skips the quantum step entirely) - this is the real "with vs without the quantum layer" comparison for this architecture
def collectActivations(model, params, evalPoints):
    preLayerOut, postQuantumOut, outputs = [], [], []
    isQuantum = hasattr(model, "quantum_circuit")

    for t, x in evalPoints:
        angles = model.pre_layer(t, x, params)
        preLayerOut.append(np.array(angles))

        if isQuantum:
            W_q = params[2]
            postQuantum = model.quantum_circuit(angles, W_q)
        else:
            postQuantum = angles #nothing sits between pre-layer and post-layer in the classical control
        postQuantumOut.append(np.array(postQuantum))

        outputs.append(float(model.network(t, x, params)))

    return np.array(preLayerOut), np.array(postQuantumOut), np.array(outputs)

def buildEvalPoints(nx=20, nt=20):
    xs, ts = buildEvalGrid(nx, nt)
    return [(t, x) for t in ts for x in xs]

def perNeuronStats(activations):
    activations = np.array(activations)
    if activations.ndim == 1:
        activations = activations.reshape(-1, 1)
    return np.mean(activations, axis=0), np.std(activations, axis=0)

def loadModelAndParams(kind, checkpointPath):
    if kind == "classical":
        params, _ = loadCheckpoint(checkpointPath)
        return main_classical, params

    params, config = loadCheckpoint(checkpointPath)
    model = main.build_model(n_qubits=config["n_qubits"], n_reuploads=config["n_reuploads"])
    return model, params

def runActivationAnalysis(classicalCheckpoint, quantumCheckpoint, nx=20, nt=20, resultsPath=None, plotPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "activation_analysis.csv")
    plotPath = plotPath or os.path.join(RESULTS_DIR, "activation_comparison.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    evalPoints = buildEvalPoints(nx, nt)

    rows = []
    postLayerByModel = {}

    for label, kind, checkpointPath in [("classical_control", "classical", classicalCheckpoint),
                                          ("quantum_pinn", "quantum", quantumCheckpoint)]:
        model, params = loadModelAndParams(kind, checkpointPath)
        preLayerOut, postQuantumOut, outputs = collectActivations(model, params, evalPoints)

        preMean, preStd = perNeuronStats(preLayerOut)
        postMean, postStd = perNeuronStats(postQuantumOut)
        outMean, outStd = perNeuronStats(outputs)

        for i in range(len(preMean)):
            rows.append({"model": label, "layer": "pre_layer_encoding", "neuron": i,
                         "mean_activation": float(preMean[i]), "std_activation": float(preStd[i])})
        for i in range(len(postMean)):
            rows.append({"model": label, "layer": "post_quantum_layer", "neuron": i,
                         "mean_activation": float(postMean[i]), "std_activation": float(postStd[i])})
        rows.append({"model": label, "layer": "output", "neuron": 0,
                     "mean_activation": float(outMean[0]), "std_activation": float(outStd[0])})

        postLayerByModel[label] = (postMean, postStd)
        print(f"{label}: post-quantum-layer mean+/-std per neuron = "
              f"{[f'{float(m):.4f}+/-{float(s):.4f}' for m, s in zip(postMean, postStd)]}", flush=True)

    with open(resultsPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layer", "neuron", "mean_activation", "std_activation"])
        writer.writeheader()
        writer.writerows(rows)

    #one subplot per model rather than a shared bar chart - the classical control's hidden width (main_classical.N_Qubits) and the quantum model's n_qubits don't have to match (e.g. a 4-unit classical control vs a checkpoint trained at n_qubits=3), so there's no shared x-axis to put them on side by side, but a shared y-axis still makes the activation scales directly comparable
    labels = list(postLayerByModel.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        mean, std = postLayerByModel[label]
        xPos = np.arange(len(mean))
        ax.bar(xPos, mean, 0.6, yerr=std, capsize=3)
        ax.set_xlabel("neuron index")
        ax.set_title(label)
        ax.set_xticks(xPos)

    axes[0].set_ylabel("activation (mean +/- std)")
    fig.suptitle("post-quantum-layer activations: classical control vs QAPINN")
    plt.tight_layout()
    plt.savefig(plotPath)
    plt.close()

    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-checkpoint", required=True)
    parser.add_argument("--quantum-checkpoint", required=True)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--nt", type=int, default=20)
    args = parser.parse_args()
    runActivationAnalysis(args.classical_checkpoint, args.quantum_checkpoint, nx=args.nx, nt=args.nt)
