**Version**: v1.0

_Built for the WISER Global Quantum+AI Program 2026 - BQP Industry Challenge_

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/JitheshMithra/SPECops/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
![Field](https://img.shields.io/badge/Field-QML-purple) 
<p align="center">
  <a href="https://www.qinetic.org/">
    <img width="769" height="279" alt="image" src="https://github.com/user-attachments/assets/45ade68c-9838-4e06-aaea-3efa7e4adb11" />
  </a>
</p>

<p align="center"><strong>Spectral Explainability & Circuit Optimization for Quantum-Assisted Physics-Informed Neural Networks</strong></p>

SPECops starts from a simple question about quantum-assisted PINNs: when you swap a classical layer for a quantum circuit and the network gets more accurate, is that actually because the quantum layer is more expressive, or is something else going on? Most benchmarks stop at "it worked" or "it didn't." This project tries to get underneath that, using the viscous Burgers' equation as a testbed.

This sweeps qubit count and data re-upload count across a quantum-assisted PINN, works out each configuration's accessible Fourier frequency range from first principles, and checks whether that range actually predicts where the network struggles. Along the way, the quantum layer turns out to collapse into producing nearly identical output across all its qubits when it's under-resourced; a real, measurable signature of the expressivity ceiling, not just a side effect. There's also a finding that a chunk of the apparent "quantum advantage" was really just faster convergence under a short training budget, which only showed up once training ran longer. Runs entirely on a simulator, nothing here needs quantum hardware.

**What it does:**
- Trains classical PINNs and QAPINNs on 1D viscous Burgers' equation against an exact Cole-Hopf reference solution
- Sweeps qubit count (3, 4, 5) x data re-upload count (1, 2, 3, 5), multi-seed
- Computes each configuration's accessible Fourier frequency spectrum (qml.fourier, theta-space and physical-x) and checks it against the target solution's own frequency content, region by region
- Measures gradient variance across random initializations as a barren-plateau check
- Measures representational redundancy inside the trained quantum layer (pairwise correlation between measured-qubit outputs) as a mechanistic account of the expressivity ceiling, not just a correlation
- Compares classical and quantum architectures under matched parameter counts and matched optimizer/epoch budgets
- Exports results as CSV/JSON/PNG under results/
