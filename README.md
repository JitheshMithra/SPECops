**Version**: v2.5

_Built for the WISER Global Quantum+AI Program 2026 - [BQP Industry Challenge](https://docs.google.com/document/d/1X4xGUgML3F0ZKSTy7G4qfpRp6xlghJrEY8CtPVjH-4o/edit?tab=t.0)_

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

This sweeps qubit count and data re-upload count across a quantum-assisted PINN, works out each configuration's accessible Fourier frequency range from first principles, and checks whether that range actually predicts where the network struggles. Runs entirely on Pennylane simulator, nothing here needs quantum hardware. Spectral bias, the tendency of neural networks to fit low-frequency components before high-frequency ones, is the named phenomenon this project provides a mechanistic account of, connecting it directly to the Fourier ceiling imposed by circuit structure. 

**What it does:**
- Trains classical PINNs and QAPINNs on 1D viscous Burgers' equation against an exact Cole-Hopf reference solution
- Sweeps qubit count (3, 4, 5) x data re-upload count (1, 2, 3, 5), multi-seed
- Computes each configuration's accessible Fourier frequency spectrum (qml.fourier, theta-space and physical-x) and checks it against the target solution's own frequency content, region by region
- Measures gradient variance across random initializations as a barren-plateau check
- Measures representational redundancy inside the trained quantum layer (pairwise correlation and SVD effective rank of post-quantum-layer outputs,  a linear algebra tool for quantifying representational redundancy instead of just measuring correlation) as a mechanistic account of the expressivity ceiling, not just a correlation
- Compares classical and quantum architectures under matched parameter counts and matched optimizer/epoch budgets
- Exports results as CSV/JSON/PNG under ```results/```

## Technical Report

The full writeup, methodology, results, and the honest caveats (single-seed extended runs, an under-parameterized early baseline, etc.) will be posted soon

## Getting Started

### Installation

```bash
git clone https://github.com/JitheshMithra/SPECops/
cd SPECops/src
pip install -r requirements.txt
```

### Running

All scripts run from the `src` directory.

**Classical control (run this first, it's the sanity check):**
```bash
python main_classical.py
```

**Single QAPINN run:**
```bash
python main.py
```

**Evaluate against the Cole-Hopf reference:**
```bash
python eval.py
```

**Full sweep (12 configs x 3 seeds):**
```bash
python sweep.py
```
Crash-durable: writes a row per seed as it finishes, so a killed run can resume instead of restarting from scratch.

**Fourier / expressivity analysis (no training needed, purely structural):**
```bash
python fourier_spectrum.py
python shock_spectrum.py
python frequency_unit_conversion.py
```

**Trainability and redundancy diagnostics:**
```bash
python gradient_variance.py
python activation_analysis.py
python activation_diversity.py
python flatness_check.py
```

**Extended training / optimizer comparisons:**
```bash
python extend_training.py --n-qubits 5 --n-reuploads 5 --seed 0 --checkpoint sweep_checkpoints/q5_r5_s0.pkl --additional-epochs 400
python classical_deep_baseline.py --hidden-layers 8 --hidden-width 20 --epochs 5000 --lr 1e-3
python train_adam.py --n-qubits 5 --n-reuploads 5 --epochs 1000 --lr 1e-3
python run_classical_comparison.py
```
### Results 

| File | Produced by | What it contains |
|---|---|---|
| `sweep_results.csv` / `sweep_results_stratified.csv` | `sweep.py` | Per-seed accuracy/timing/params; split by smooth vs. shock region |
| `fourier_spectrum_results.csv` | `fourier_spectrum.py` | Accessible Fourier frequencies per config (structural, no training needed) |
| `burgers_shock_spectrum.json` | `shock_spectrum.py` | Target solution's required bandwidth, by time snapshot |
| `frequency_unit_conversion.csv`, `expressivity_vs_shock.csv` | `frequency_unit_conversion.py` | Theta-space ceiling converted to physical-x units; sufficiency-ratio analysis |
| `gradient_variance_results.csv` | `gradient_variance.py` | Barren-plateau check across configs |
| `activation_diversity.csv`, `activation_measurement_check.csv` | `activation_diversity.py` | The redundancy metric; check for whether it's a measurement artifact |
| `flatness_check.csv/.png` | `flatness_check.py` | Solution flatness vs. reference: the spectral-bias confirmation |
| `classical_comparison.csv`, `longer_training_results.csv` | comparison scripts | Classical vs. quantum, matched-parameter and matched-optimizer results |
| `results/invalid_pre_fix/` | - | Archived, invalid: predates an optimizer bug fix. Don't use anything in here. |

## Limitations

- Only one PDE so far (viscous Burgers'); a heat-equation cross-check is scaffolded but not finished
- Extended-training comparisons past the base sweep are currently single-seed
- The original classical control was noticeably smaller (parameter count) than the largest QAPINN tested; a parameter-matched control exists but the optimizer-matched comparison is still being finalized
- No configuration tested actually reaches expressivity sufficiency in either the smooth or shock region of the solution; read the sufficiency numbers as relative distance, not a pass/fail line
- Simulator only, nothing here has touched real quantum hardware

## Future Work

- Pull the PDE-specific logic behind a shared interface so adding a new PDE (starting with the heat equation) doesn't mean touching the core pipeline
- Build the direct CLI input for parameters from users, a tool to observe the effect assumptions have on behavior
- Track redundancy/effective rank across training epochs instead of just at fixed checkpoints
- Compare entanglement patterns and measurement operators (expectation values vs. probability vector)
- Shapley-value gate attribution for the variational block

### Acknowledgements
This project was carried out under [**Qinetic Labs**](https://www.qinetic.org/), with Jithesh Mithra as head researcher, leading the project and responsible for the majority of the implementation, experiments, and analysis. Thanks to Isaac Leon for contributions on the theory side, including work on the Fourier-ceiling framing and the disentanglement analysis.

**Contact**:
- _Email_: jitheshmithra412 [at] gmail [dot] com **and/or** isaacleon0907 [at] gmail [dot] com
- _LinkedIn_: https://www.linkedin.com/in/jitheshmithra/
