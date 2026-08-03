import csv
import os

from pennylane import numpy as np
from pennylane import fourier

import main

DEFAULT_QUBITS = (3, 4, 5)
DEFAULT_REUPLOADS = (1, 2, 3, 5)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#the accessible frequency spectrum of a data-reuploading circuit is a static property of its structure (how many times each input gets fed into a rotation gate) - it doesn't depend on trained weights, so this runs standalone, no checkpoint or training needed
def computeSpectrum(nQubits, nReuploads):
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)

    dummyInputs = np.zeros(nQubits, requires_grad=True)
    dummyWeights = np.zeros((nReuploads, nQubits, 2), requires_grad=True)

    spectrumFn = fourier.qnode_spectrum(model.quantum_circuit, argnum=[0])
    return spectrumFn(dummyInputs, dummyWeights)

#every qubit gets the identical encoding + reupload treatment, so every input component ends up with the same frequency list - just report one representative list plus its max, instead of repeating it n_qubits times
def spectrumSummary(spectrum):
    freqLists = [freqs for freqsByIndex in spectrum.values() for freqs in freqsByIndex.values()]
    representative = freqLists[0]
    maxFrequency = max(abs(f) for f in representative)
    return representative, maxFrequency

#saves one row per config so this can be joined with sweep.py's accuracy results and gradient_variance.py's barren-plateau results on (n_qubits, n_reuploads)
def runSpectrumSweep(nQubitsList=DEFAULT_QUBITS, nReuploadsList=DEFAULT_REUPLOADS, resultsPath=None):
    resultsPath = resultsPath or os.path.join(RESULTS_DIR, "fourier_spectrum_results.csv")
    os.makedirs(os.path.dirname(os.path.abspath(resultsPath)), exist_ok=True)
    fileExists = os.path.exists(resultsPath)
    with open(resultsPath, "a", newline="") as f:
        writer = csv.writer(f)
        if not fileExists:
            writer.writerow(["n_qubits", "n_reuploads", "max_frequency", "frequencies"])

        for nQubits in nQubitsList:
            for nReuploads in nReuploadsList:
                spectrum = computeSpectrum(nQubits, nReuploads)
                freqs, maxFreq = spectrumSummary(spectrum)
                freqStr = ";".join(str(freq) for freq in freqs)

                print(f"n_qubits={nQubits} n_reuploads={nReuploads} max_frequency={maxFreq}")
                writer.writerow([nQubits, nReuploads, maxFreq, freqStr])
                f.flush()

if __name__ == "__main__":
    runSpectrumSweep()
