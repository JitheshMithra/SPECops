from pennylane import numpy as np
from pennylane import fourier

import main

#the accessible frequency spectrum of a data-reuploading circuit is a static property of its structure (how many times each input gets fed into a rotation gate) - it doesn't depend on trained weights, so this runs standalone, no checkpoint or training needed
def computeSpectrum(nQubits, nReuploads):
    model = main.build_model(n_qubits=nQubits, n_reuploads=nReuploads)

    dummyInputs = np.zeros(nQubits, requires_grad=True)
    dummyWeights = np.zeros((nReuploads, nQubits, 2), requires_grad=True)

    spectrumFn = fourier.qnode_spectrum(model.quantum_circuit, argnum=[0])
    return spectrumFn(dummyInputs, dummyWeights)

if __name__ == "__main__":
    nQubitsList = (3, 4, 5)
    nReuploadsList = (1, 2, 3, 5)

    for nQubits in nQubitsList:
        for nReuploads in nReuploadsList:
            spectrum = computeSpectrum(nQubits, nReuploads)
            print(f"n_qubits={nQubits} n_reuploads={nReuploads}")
            for inputName, freqsByIndex in spectrum.items():
                for idx, freqs in freqsByIndex.items():
                    print(f"  {inputName}{list(idx)}: {freqs}")
