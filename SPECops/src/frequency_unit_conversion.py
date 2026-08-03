import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qp
from pennylane import numpy as np

import main

DEFAULT_QUBITS = (3, 4, 5)
DEFAULT_REUPLOADS = (1, 2, 3, 5)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#the pre-layer takes (t,x) jointly (see main.py's pre_layer()), so d(theta)/dx technically depends on t too - fixing t at the domain midpoint here rather than pretending theta is purely a function of x, and flagging that simplification
REPRESENTATIVE_T = 0.5

#pre_layer's weights (W1, b1) only depend on n_qubits (see main.py's init_params: W1 has shape (n_qubits, 2)), not on n_reuploads - the quantum layer weights W_q are what depend on n_reuploads, and those don't feed dtheta/dx at all, so for a clean structural comparison across n_reuploads at fixed n_qubits, W1/b1 must be IDENTICAL across that comparison; previously each row called model.init_params() with whatever state the global RNG happened to be in, so W1 differed randomly row-to-row even at fixed n_qubits, and that noise alone was enough to make max_frequency_x_effective non-monotonic in n_reuploads despite the theta-space ceiling being cleanly monotonic - reseeding with the same fixed seed before every init_params() call (W1 is drawn first inside init_params, before anything reupload-count-dependent) pins W1/b1 to one fixed value per n_qubits regardless of n_reuploads
PRE_LAYER_SEED = 12345

#note on method: the task asked for torch.autograd, but this project has no torch dependency anywhere - it's built on pennylane's own autodiff (qp.grad / qp.jacobian), which is exactly what pde_residual() already uses for u_t/u_x/u_xx, and using that instead accomplishes the same goal (differentiate the *actual* pre_layer() code instead of hand-deriving the tanh chain rule) with the tool this codebase is actually built on
def dThetaDx(model, params, t, xValues):
    grads = []
    for x in xValues:
        xVar = np.array(x, requires_grad=True)
        thetaOfX = lambda x_: model.pre_layer(t, x_, params)
        grads.append(qp.jacobian(thetaOfX)(xVar))
    return np.array(grads) #shape (len(xValues), n_qubits)

def analyzeScaling(model, params, nx=500, t=REPRESENTATIVE_T):
    xValues = np.linspace(main.X_Min, main.X_Max, nx)
    grads = dThetaDx(model, params, t, xValues)
    magnitude = np.abs(grads)

    meanSlope = float(np.mean(magnitude))
    minSlope = float(np.min(magnitude))
    maxSlope = float(np.max(magnitude))
    #if the spread is small relative to the mean, a single conversion factor is a fair stand-in
    nearConstant = (maxSlope - minSlope) < 0.2 * meanSlope

    return xValues, grads, meanSlope, minSlope, maxSlope, nearConstant

def plotScaling(xValues, grads, path):
    plt.figure(figsize=(7, 4))
    for q in range(grads.shape[1]):
        plt.plot(xValues, grads[:, q], label=f"qubit {q}")
    plt.xlabel("x")
    plt.ylabel("d(theta)/dx")
    plt.title(f"pre-layer slope vs x (t={REPRESENTATIVE_T})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def loadSpectrumResults(path):
    with open(path, newline="") as f:
        return [{"n_qubits": int(row["n_qubits"]), "n_reuploads": int(row["n_reuploads"]),
                  "max_frequency_theta": float(row["max_frequency"])}
                for row in csv.DictReader(f)]

def loadShockBandwidths(path):
    with open(path) as f:
        payload = json.load(f)
    return [(snap["t"], snap["effective_bandwidth_99pct_freq"]) for snap in payload["snapshots"]]

def run(spectrumCsv=None, shockJson=None, outCsv=None, comparisonCsv=None, plotPath=None):
    spectrumCsv = spectrumCsv or os.path.join(RESULTS_DIR, "fourier_spectrum_results.csv")
    shockJson = shockJson or os.path.join(RESULTS_DIR, "burgers_shock_spectrum.json")
    outCsv = outCsv or os.path.join(RESULTS_DIR, "frequency_unit_conversion.csv")
    comparisonCsv = comparisonCsv or os.path.join(RESULTS_DIR, "expressivity_vs_shock.csv")
    plotPath = plotPath or os.path.join(RESULTS_DIR, "pre_layer_slope_vs_x.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    spectrumRows = loadSpectrumResults(spectrumCsv)
    bandwidths = loadShockBandwidths(shockJson)

    conversionRows = []
    plotted = False
    for row in spectrumRows:
        nQubits, nReuploads = row["n_qubits"], row["n_reuploads"]
        model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)
        np.random.seed(PRE_LAYER_SEED) #same seed regardless of n_reuploads -> identical W1/b1 for a given n_qubits
        params = model.init_params()

        xValues, grads, meanSlope, minSlope, maxSlope, nearConstant = analyzeScaling(model, params)

        if not plotted: #one representative plot is enough - the linear-vs-saturating check is qualitative, not per-config
            plotScaling(xValues, grads, plotPath)
            plotted = True

        maxFreqTheta = row["max_frequency_theta"]
        #freq_x = freq_theta * d(theta)/dx via the chain rule - see note above, this is a multiply, not a divide (dividing would invert the relationship: a small dtheta/dx means theta barely moves as x moves, so the SAME theta-frequency covers a much LOWER x-frequency, not a higher one)
        conversionRows.append({
            "n_qubits": nQubits, "n_reuploads": nReuploads,
            "max_frequency_theta": maxFreqTheta,
            "dtheta_dx_mean": meanSlope, "dtheta_dx_min": minSlope, "dtheta_dx_max": maxSlope,
            "near_constant_scaling": nearConstant,
            "max_frequency_x_mean": maxFreqTheta * meanSlope,
            "max_frequency_x_min": maxFreqTheta * minSlope,
            "max_frequency_x_max": maxFreqTheta * maxSlope,
        })

        regime = "near-linear" if nearConstant else "tanh saturating somewhere in [-1,1]"
        print(f"n_qubits={nQubits} n_reuploads={nReuploads}: dtheta/dx mean={meanSlope:.4f} (range {minSlope:.4f}-{maxSlope:.4f}) -> {regime}")

    with open(outCsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(conversionRows[0].keys()))
        writer.writeheader()
        writer.writerows(conversionRows)

    comparisonRows = []
    for row in conversionRows:
        for t, requiredBandwidth in bandwidths:
            sufficient = row["max_frequency_x_mean"] >= requiredBandwidth
            comparisonRows.append({
                "n_qubits": row["n_qubits"], "n_reuploads": row["n_reuploads"],
                "t": t, "required_bandwidth_x": requiredBandwidth,
                "max_frequency_x_effective": row["max_frequency_x_mean"],
                "expressivity_sufficient": sufficient,
            })
            print(f"  q{row['n_qubits']}_r{row['n_reuploads']} @ t={t}: ceiling={row['max_frequency_x_mean']:.3f} vs required={requiredBandwidth:.2f} -> sufficient={sufficient}")

    with open(comparisonCsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisonRows[0].keys()))
        writer.writeheader()
        writer.writerows(comparisonRows)

    return conversionRows, comparisonRows

if __name__ == "__main__":
    run()
