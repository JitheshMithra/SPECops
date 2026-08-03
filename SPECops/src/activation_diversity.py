import csv
import os

import pennylane as qp
from pennylane import numpy as np

import main
import main_classical
import sweep
from checkpoint import loadCheckpoint
from activation_analysis import buildEvalPoints, perNeuronStats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#alternate readout: mixed observables (Z, X, Y cycling per wire) instead of PauliZ on every wire - same gate sequence as main.py's quantum_circuit (copied deliberately rather than imported, since build_model()'s qnode has PauliZ baked into its return statement with no hook to swap it), only the measurement basis differs, so if post-quantum-layer uniformity is a measurement artifact (e.g. PauliZ saturating similarly on every wire because of how the entangling ring mixes things), varying the observable per wire should break the uniformity - if it persists, the uniformity is a property of the underlying state, not the measurement
MIXED_OBSERVABLES = [qp.PauliZ, qp.PauliX, qp.PauliY]

def buildMixedObservableCircuit(n_qubits, n_reuploads):
    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit(inputs, weights):
        for layer in range(n_reuploads):
            for q in range(n_qubits):
                qp.RY(inputs[q], wires=q)
            for q in range(n_qubits):
                qp.RY(weights[layer, q, 0], wires=q)
                qp.RZ(weights[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qp.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qp.expval(MIXED_OBSERVABLES[i % 3](i)) for i in range(n_qubits)]

    return circuit

#same extraction as activation_analysis.collectActivations, but returns raw per-point activation matrices (not just mean/std) since the diversity metric needs the full matrix to compute pairwise correlation across neurons
def collectActivationMatrices(model, params, evalPoints, quantumCircuitOverride=None):
    preLayerOut, postOut, outputs = [], [], []
    isQuantum = hasattr(model, "quantum_circuit")

    for t, x in evalPoints:
        angles = model.pre_layer(t, x, params)
        preLayerOut.append(np.array(angles))

        if isQuantum:
            W_q = params[2]
            circuitFn = quantumCircuitOverride or model.quantum_circuit
            post = circuitFn(angles, W_q)
        else:
            post = angles
        postOut.append(np.array(post))

        outputs.append(float(model.network(t, x, params)))

    return np.array(preLayerOut), np.array(postOut), np.array(outputs)

#across-neuron diversity: how much do neurons differ from each other over the same eval batch - two complementary views: across_neuron_variance is the variance of the per-neuron MEANS (low = neurons are all centered in the same place on average), mean_pairwise_correlation is the average Pearson correlation between every pair of neurons' activation vectors across the eval batch (high = neurons move together / carry redundant information, low = they carry distinct information even if their means happen to coincide)
def diversityMetrics(activations):
    activations = np.array(activations, dtype=float)
    if activations.ndim == 1 or activations.shape[1] < 2:
        return float(np.var(np.mean(activations.reshape(len(activations), -1), axis=0))), None

    neuronMeans = np.mean(activations, axis=0)
    acrossNeuronVariance = float(np.var(neuronMeans))

    corrMatrix = np.corrcoef(activations, rowvar=False)
    n = corrMatrix.shape[0]
    offDiag = [corrMatrix[i, j] for i in range(n) for j in range(n) if i != j]
    meanPairwiseCorrelation = float(np.mean(offDiag)) if offDiag else None

    return acrossNeuronVariance, meanPairwiseCorrelation

def loadModelAndParams(kind, checkpointPath):
    if kind == "classical":
        params, _ = loadCheckpoint(checkpointPath)
        return main_classical, params
    params, config = loadCheckpoint(checkpointPath)
    model = main.build_model(n_qubits=config["n_qubits"], n_reuploads=config["n_reuploads"])
    return model, params

# part 1: measurement-artifact check
def checkMeasurementArtifact(configs, nx=20, nt=20, outPath=None):
    outPath = outPath or os.path.join(RESULTS_DIR, "activation_measurement_check.csv")
    evalPoints = buildEvalPoints(nx, nt)

    rows = []
    for label, checkpointPath in configs:
        model, params = loadModelAndParams("quantum", checkpointPath)
        nQubits = model.n_qubits

        _, postSameObs, _ = collectActivationMatrices(model, params, evalPoints)
        meanSame, _ = perNeuronStats(postSameObs)

        mixedCircuit = buildMixedObservableCircuit(nQubits, model.n_reuploads)
        _, postMixedObs, _ = collectActivationMatrices(model, params, evalPoints, quantumCircuitOverride=mixedCircuit)
        meanMixed, _ = perNeuronStats(postMixedObs)

        spreadSame = float(np.std(meanSame)) #std ACROSS neuron means - the "how uniform" number
        spreadMixed = float(np.std(meanMixed))

        print(f"{label}: same-observable (PauliZ x{nQubits}) neuron means = {[f'{float(m):.4f}' for m in meanSame]}, spread(std across means)={spreadSame:.4f}", flush=True)
        print(f"{label}: mixed-observable (Z/X/Y cycling) neuron means   = {[f'{float(m):.4f}' for m in meanMixed]}, spread(std across means)={spreadMixed:.4f}", flush=True)

        rows.append({"config": label, "n_qubits": nQubits,
                      "same_observable_means": ";".join(f"{float(m):.6f}" for m in meanSame),
                      "same_observable_spread": spreadSame,
                      "mixed_observable_means": ";".join(f"{float(m):.6f}" for m in meanMixed),
                      "mixed_observable_spread": spreadMixed})

    with open(outPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {outPath}", flush=True)
    return rows

#part 2+3: multi-config activation analysis + diversity metrics
def runExpandedAnalysis(configs, nx=20, nt=20, activationCsvPath=None, diversityCsvPath=None):
    activationCsvPath = activationCsvPath or os.path.join(RESULTS_DIR, "activation_analysis.csv")
    diversityCsvPath = diversityCsvPath or os.path.join(RESULTS_DIR, "activation_diversity.csv")
    evalPoints = buildEvalPoints(nx, nt)

    activationRows = []
    diversityRows = []

    for label, kind, checkpointPath in configs:
        model, params = loadModelAndParams(kind, checkpointPath)
        preLayerOut, postOut, outputs = collectActivationMatrices(model, params, evalPoints)

        preMean, preStd = perNeuronStats(preLayerOut)
        postMean, postStd = perNeuronStats(postOut)
        outMean, outStd = perNeuronStats(outputs)

        for i in range(len(preMean)):
            activationRows.append({"config": label, "layer": "pre_layer_encoding", "neuron": i,
                                    "mean_activation": float(preMean[i]), "std_activation": float(preStd[i])})
        for i in range(len(postMean)):
            activationRows.append({"config": label, "layer": "post_quantum_layer", "neuron": i,
                                    "mean_activation": float(postMean[i]), "std_activation": float(postStd[i])})
        activationRows.append({"config": label, "layer": "output", "neuron": 0,
                                "mean_activation": float(outMean[0]), "std_activation": float(outStd[0])})

        for layerName, activations in [("pre_layer_encoding", preLayerOut), ("post_quantum_layer", postOut)]:
            acrossVar, meanCorr = diversityMetrics(activations)
            diversityRows.append({"model": kind, "config": label, "layer": layerName,
                                   "across_neuron_variance": acrossVar,
                                   "mean_pairwise_correlation": meanCorr})
            print(f"{label} / {layerName}: across_neuron_variance={acrossVar:.6f} mean_pairwise_correlation={meanCorr}", flush=True)

    with open(activationCsvPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "layer", "neuron", "mean_activation", "std_activation"])
        writer.writeheader()
        writer.writerows(activationRows)
    print(f"saved {activationCsvPath}", flush=True)

    with open(diversityCsvPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "config", "layer", "across_neuron_variance", "mean_pairwise_correlation"])
        writer.writeheader()
        writer.writerows(diversityRows)
    print(f"saved {diversityCsvPath}", flush=True)

    return activationRows, diversityRows

if __name__ == "__main__":
    classicalCkpt = os.path.join(sweep.DEFAULT_CHECKPOINT_DIR, "classical_s0.pkl")
    q3r1Ckpt = sweep.checkpointPathFor(3, 1, sweep.DEFAULT_CHECKPOINT_DIR, seed=0)
    q5r5Ckpt = sweep.checkpointPathFor(5, 5, sweep.DEFAULT_CHECKPOINT_DIR, seed=0)

    print("measurement-artifact check (same-observable vs mixed-observable readout)", flush=True)
    checkMeasurementArtifact([("q3_r1", q3r1Ckpt), ("q5_r5", q5r5Ckpt)])

    print("expanded activation analysis (classical, q3_r1, q5_r5)", flush=True)
    runExpandedAnalysis([
        ("classical_control", "classical", classicalCkpt),
        ("q3_r1", "quantum", q3r1Ckpt),
        ("q5_r5", "quantum", q5r5Ckpt),
    ])
