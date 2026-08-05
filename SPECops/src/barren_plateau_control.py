import argparse
import csv
import json
import os

import pennylane as qp
from pennylane import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#gradient_variance.py measures whether OUR ansatz sits on a barren plateau. this script measures whether that measurement would notice one if it were there. it runs the same estimator protocol (N random inits -> variance of one fixed gradient component) on the random parameterised circuit family from McClean et al. 2018 (arXiv:1803.11173), which is the family the plateau is proven for, and checks the fitted decay against their published slope. without this, a flat gradient_variance_results.csv is ambiguous between "no plateau here" and "estimator too blunt to see one" - with it, the flat result becomes a claim
DEFAULT_CONTROL_QUBITS = (2, 4, 6, 8)

#sweep.py's own grid, so the control can also be reported at exactly the sizes our QAPINN runs at
SWEEP_QUBITS = (3, 4, 5)

#published natural-log slopes of Var[d_theta E] vs n_qubits. McClean et al. plot semilog with ln on the y-axis, not log10 - Fig 5 (H = |0..0><0..0|, their Appendix II) and Fig 3 (H = Z1 Z2, the two-local Pauli term). converting is the single easiest place to be off by a factor of 2.303 and silently "fail" a validation that actually passed
MCCLEAN_REFERENCE_SLOPE_LN = {"projector_zero": -1.374, "zz": -0.694}

#McClean et al. run their numerics with the layer count a modest linear function of the qubit count - at fixed shallow depth U_- is nearly trivial and the second moment hasn't converged, so the decay doesn't show and the control would fail for a reason that has nothing to do with our estimator
DEFAULT_LAYERS_PER_QUBIT = 2


#the RPQC of McClean et al. Fig 2: sqrt(Hadamard) = RY(pi/4) on every wire so no Pauli axis is preferred, then nLayers blocks of one uniformly random Pauli rotation per wire followed by a 1D CZ ladder. paulis is passed in rather than sampled inside so the caller controls the RNG and a run is reproducible from its seed
def rpqcCircuit(nQubits, nLayers, paulis, observable="projector_zero"):
    dev = qp.device("default.qubit", wires=nQubits)
    gateOf = {"X": qp.RX, "Y": qp.RY, "Z": qp.RZ}

    @qp.qnode(dev)
    def circuit(theta):
        for q in range(nQubits):
            qp.RY(np.pi / 4, wires=q)
        for layer in range(nLayers):
            for q in range(nQubits):
                gateOf[paulis[layer][q]](theta[layer, q], wires=q)
            for q in range(nQubits - 1):
                qp.CZ(wires=[q, q + 1])
        if observable == "zz":
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(min(1, nQubits - 1)))
        return qp.expval(qp.Projector(np.zeros(nQubits, dtype=int), wires=range(nQubits)))

    return circuit


#same protocol as gradientVariance() in gradient_variance.py - draw nSamples independent uniform inits, take one fixed gradient component at each, return its sample variance. the component is theta[0,0] (first layer, first wire) to match McClean's theta_{1,1}; picking the LAST parameter instead gives an exactly-zero gradient for the ZZ observable, since a final-layer rotation on a wire the observable doesn't touch can't move it, which reads as a spectacular plateau and is really just a bad choice of component
def controlGradientVariance(nQubits, nLayers, nSamples=200, observable="projector_zero", seed=0):
    np.random.seed(seed)
    grads = []
    for _ in range(nSamples):
        paulis = [[str(np.random.choice(["X", "Y", "Z"])) for _ in range(nQubits)]
                  for _ in range(nLayers)]
        circuit = rpqcCircuit(nQubits, nLayers, paulis, observable=observable)
        theta = np.array(np.random.uniform(0, 2 * np.pi, size=(nLayers, nQubits)), requires_grad=True)
        grads.append(float(qp.grad(circuit)(theta)[0, 0]))
    grads = np.array(grads)
    return float(np.var(grads)), float(np.mean(grads))


#plain least-squares fit of ln(Var) against qubit count, written out rather than calling polyfit so it works on pennylane's autograd-wrapped numpy without thinking about it. returns the slope in ln units to be directly comparable to the published number, and r2 so a fit that happens to have the right slope through noise doesn't pass
def fitLogSlope(xs, variances):
    pairs = [(float(x), float(v)) for x, v in zip(xs, variances) if v > 0]
    if len(pairs) < 2:
        return {"slope_ln": float("nan"), "intercept_ln": float("nan"),
                "r_squared": float("nan"), "n_points": len(pairs)}
    xs = np.array([p[0] for p in pairs])
    ys = np.log(np.array([p[1] for p in pairs]))
    xBar, yBar = float(np.mean(xs)), float(np.mean(ys))
    sxx = float(np.sum((xs - xBar) ** 2))
    slope = float(np.sum((xs - xBar) * (ys - yBar)) / sxx)
    intercept = yBar - slope * xBar
    resid = ys - (slope * xs + intercept)
    ssTot = float(np.sum((ys - yBar) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ssTot if ssTot > 0 else float("nan")
    return {"slope_ln": slope, "intercept_ln": float(intercept),
            "r_squared": r2, "n_points": len(pairs)}


#no subprocess isolation here, unlike gradient_variance.py. that script needs it because pde_residual already nests two levels of autodiff internally and differentiating it again w.r.t. W_q makes three, which is what blew out memory mid-sweep. this one differentiates a bare expectation value once - a single level, no pde_residual anywhere - so the graph never gets large enough to be worth the subprocess overhead
def runControlSweep(qubitList=DEFAULT_CONTROL_QUBITS, layersPerQubit=DEFAULT_LAYERS_PER_QUBIT,
                    nSamples=200, observable="projector_zero", seed=0, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "barren_plateau_control.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    variances = []
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_layers", "observable", "n_samples",
                             "grad_variance", "grad_mean"])

        for nQubits in qubitList:
            nLayers = layersPerQubit * nQubits
            try:
                variance, mean = controlGradientVariance(
                    nQubits, nLayers, nSamples=nSamples, observable=observable, seed=seed + nQubits)
            except MemoryError:
                #same retry-once convention the rest of the repo settled on, though this script is
                #cheap enough that it has never actually tripped it
                print(f"n_qubits={nQubits} MemoryError, retrying once", flush=True)
                variance, mean = controlGradientVariance(
                    nQubits, nLayers, nSamples=nSamples, observable=observable, seed=seed + nQubits)

            variances.append(variance)
            print(f"n_qubits={nQubits} n_layers={nLayers} grad_variance={variance:.3e}", flush=True)
            writer.writerow([nQubits, nLayers, observable, nSamples, variance, mean])
            f.flush()

    fit = fitLogSlope(qubitList, variances)
    reference = MCCLEAN_REFERENCE_SLOPE_LN[observable]
    fit.update({
        "observable": observable,
        "qubit_counts": list(qubitList),
        "layers_per_qubit": layersPerQubit,
        "n_samples": nSamples,
        "variances": variances,
        "reference_slope_ln": reference,
        "relative_error": abs(fit["slope_ln"] - reference) / abs(reference),
    })
    #two conditions, not one: the slope has to land near the published value AND the fit has to
    #actually be a line. a noisy scatter that happens to average out to -1.4 is not a reproduction
    fit["passed"] = bool(fit["relative_error"] < 0.25 and fit["r_squared"] > 0.95)
    return fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="validate the barren-plateau estimator against McClean et al. 2018 before trusting gradient_variance_results.csv")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="random inits per qubit count (200 is enough for a clean fit; 50 is noticeably noisier)")
    parser.add_argument("--observable", default="projector_zero", choices=["projector_zero", "zz"],
                        help="projector_zero reproduces McClean Fig 5 and is the cheaper/cleaner check; zz reproduces Fig 3 but needs far more depth than we can simulate to converge")
    parser.add_argument("--layers-per-qubit", type=int, default=DEFAULT_LAYERS_PER_QUBIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--also-sweep-grid", action="store_true",
                        help="additionally run the control at sweep.py's own qubit counts (3,4,5) so the numbers sit alongside gradient_variance_results.csv")
    args = parser.parse_args()

    fit = runControlSweep(nSamples=args.n_samples, observable=args.observable,
                          layersPerQubit=args.layers_per_qubit, seed=args.seed)

    if args.also_sweep_grid:
        runControlSweep(qubitList=SWEEP_QUBITS, nSamples=args.n_samples,
                        observable=args.observable, layersPerQubit=args.layers_per_qubit,
                        seed=args.seed)

    summaryPath = os.path.join(RESULTS_DIR, "barren_plateau_control_fit.json")
    with open(summaryPath, "w") as f:
        json.dump(fit, f, indent=2)

    print()
    print(f"fitted slope (ln)  {fit['slope_ln']:.3f}")
    print(f"published          {fit['reference_slope_ln']}  (McClean et al. 2018)")
    print(f"r^2                {fit['r_squared']:.3f}")
    print(f"relative error     {fit['relative_error']:.1%}")
    print(f"VALIDATION {'PASSED' if fit['passed'] else 'FAILED'} -> {summaryPath}")
    if not fit["passed"]:
        #loud on failure: if the protocol can't reproduce a plateau that is known to be there,
        #the flat QAPINN numbers next door don't mean what the report would say they mean
        raise SystemExit(1)
