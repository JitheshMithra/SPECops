import json
import os

from pennylane import numpy as np

import main
from eval import coleHopfU

DEFAULT_T_SNAPSHOTS = (0.1, 0.3, 0.5, 0.7, 0.9)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#looks at the actual frequency content of the burgers' shock (via FFT of the cole-hopf reference solution in x, at a few snapshots in t), so it can be compared against the circuit's accessible spectrum from fourier_spectrum.py - note: not an exact unit match, the circuit's integer frequencies are w.r.t. its own encoding angle, a nonlinear (tanh) function of the physical (t,x) inputs, not physical x directly, so treat this as a rough explainability comparison of scale, not a precise one
def spatialSpectrum(t, nx=256):
    xs = np.linspace(main.X_Min, main.X_Max, nx, endpoint=False) #periodic-style sampling for the FFT
    u = np.array([coleHopfU(x, t) for x in xs])

    spectrum = np.fft.rfft(u)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(nx, d=(xs[1] - xs[0])) #cycles per unit x

    return freqs, magnitude

#how many low frequency bins you need to capture 99% of the signal's energy - a rough stand-in for "how much bandwidth does this shock actually need"
def effectiveBandwidth(magnitude, energyThreshold=0.99):
    power = magnitude ** 2
    totalEnergy = np.sum(power)
    if totalEnergy == 0:
        return 0
    cumulative = np.cumsum(power) / totalEnergy
    return int(np.searchsorted(cumulative, energyThreshold))

def runShockSpectrum(tSnapshots=DEFAULT_T_SNAPSHOTS, nx=256, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "burgers_shock_spectrum.json")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)

    snapshots = []
    for t in tSnapshots:
        freqs, magnitude = spatialSpectrum(t, nx=nx)
        bandwidthBin = effectiveBandwidth(magnitude)

        snapshots.append({
            "t": t,
            "freqs_cycles_per_x": freqs.tolist(),
            "magnitude": magnitude.tolist(),
            "effective_bandwidth_99pct_bin": bandwidthBin,
            "effective_bandwidth_99pct_freq": float(freqs[bandwidthBin]) if bandwidthBin < len(freqs) else None,
        })
        print(f"t={t}: 99% spectral energy within bin {bandwidthBin} (~{freqs[bandwidthBin]:.2f} cycles/unit x)")

    payload = {
        "note": "circuit frequencies in fourier_spectrum_results.csv are w.r.t. the encoding angle (post tanh pre-layer), not physical x directly - compare orders of magnitude here, not exact units",
        "snapshots": snapshots,
    }
    with open(resultsPath, "w") as f:
        json.dump(payload, f, indent=2)

    return snapshots

if __name__ == "__main__":
    runShockSpectrum()
