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
- Attributes each variational layer's contribution to the trained circuit via Shapley values (q5,r5 config)
- Tracks representational redundancy via SVD effective rank, not just pairwise correlation
- CLI wrapper on `main.py` for qubit count / re-uploads / epochs / seed
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

### CLI reference

**`main.py`**: single QAPINN training run
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--n-qubits` | int | 4 | |
| `--n-reuploads` | int | 3 | |
| `--epochs` | int | 100 | |
| `--seed` | int | 0 | |
| `--checkpoint` | str | None | path to save the trained checkpoint |

**`sweep.py`**: full sweep driver
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--full-grid` | flag | off | re-eval one already-swept config's checkpoint at full 256x100 resolution instead of running the coarse sweep |
| `--n-qubits` | int | required for `--full-grid` | |
| `--n-reuploads` | int | required for `--full-grid` | |
| `--seed` | int | 0 | |
| `--nx` | int | 256 | |
| `--nt` | int | 100 | |
| `--checkpoint-dir` | str | `DEFAULT_CHECKPOINT_DIR` | |

**`eval.py`**: evaluate a checkpoint against the Cole-Hopf reference
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--checkpoint` | str | `checkpoint.pkl` | |
| `--results` | str | `results/eval_results.csv` | |
| `--model` | str | `main` | module defining `network()`/`pde_residual()` for this checkpoint, e.g. `main` or `main_classical` |
| `--nx` | int | 256 | |
| `--nt` | int | 100 | |

**`gradient_variance.py`**: barren-plateau check
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--single` | flag | off | internal, one config's variance, used by the subprocess-isolated sweep driver |
| `--n-qubits` | int | required for `--single` | |
| `--n-reuploads` | int | required for `--single` | |
| `--n-samples` | int | 50 | |
| `--batch-size` | int | 5 | |

**`activation_analysis.py`**: redundancy/activation comparison between checkpoints
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--classical-checkpoint` | str | required | |
| `--quantum-checkpoint` | str | required | |
| `--nx` | int | 20 | |
| `--nt` | int | 20 | |

**`extend_training.py`**: resume a checkpoint for extended-training / matched-optimizer comparisons
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--n-qubits` | int | required | |
| `--n-reuploads` | int | required | |
| `--seed` | int | required | |
| `--checkpoint` | str | required | existing trained checkpoint to continue from |
| `--additional-epochs` | int | required | |
| `--start-epoch` | int | 100 | epoch count the input checkpoint already represents |
| `--eval-every` | int | 100 | |
| `--lr` | float | 0.05 | |
| `--n-f-batch` | int | 20 | |
| `--eval-nx` | int | 64 | |
| `--eval-nt` | int | 50 | |
| `--out` | str | None | |
| `--out-checkpoint` | str | None | |

**`classical_deep_baseline.py`**: deeper classical control
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--hidden-layers` | int | 8 | |
| `--hidden-width` | int | 20 | |
| `--epochs` | int | 5000 | |
| `--lr` | float | 1e-3 | |
| `--n-f-batch` | int | 20 | |
| `--report-every` | int | 500 | |

**`train_adam.py`**: matched-optimizer convergence comparison
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--model` | choice | required | `quantum` or `classical_matched` |
| `--n-qubits` | int | 5 | |
| `--n-reuploads` | int | 5 | |
| `--hidden-width` | int | 18 | |
| `--epochs` | int | 5000 | |
| `--lr` | float | 1e-3 | |
| `--n-f-batch` | int | 200 | |
| `--report-every` | int | 500 | |
| `--eval-every` | int | 100 | |
| `--out-checkpoint` | str | None | |

**`heat_equation_sweep.py`**: heat-equation cross-check sweep
| Flag | Type | Default | Notes |
|---|---|---|---|
| `--n-qubits` | int | required | |
| `--n-reuploads` | int | required | |
| `--seed` | int | 0 | |
| `--epochs` | int | 100 | |
| `--n-f-batch` | int | 20 | |


**Classical control (run this first, it's the sanity check):**
```bash
python main_classical.py
```

**Single QAPINN run:**
```bash
python main.py --n-qubits 5 --n-reuploads 5 --epochs 100 --seed 0
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
python effective_rank.py
python shapley_layer_attribution.py
python heat_equation.py
python heat_equation_sweep.py
python redundancy_over_epochs.py
python plot_q5_r5_convergence.py
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
| `effective_rank.csv` | `effective_rank.py` | SVD-based effective rank per config |
| `shapley_layer_attribution.csv`, `shapley_subset_l2.csv` | `shapley_layer_attribution.py` | Per-layer Shapley attribution, q5,r5 |
| `classical_matched_adam_convergence.csv`, `q5_r5_adam_convergence.csv` | optimizer-comparison scripts | Convergence traces under matched optimizer/epochs |
| `heat_equation_sweep.csv` | `heat_equation_sweep.py` | Heat-equation cross-check sweep (in progress) |
| `redundancy_over_epochs.csv` | `redundancy_over_epochs.py` | Redundancy/effective rank tracked across training epochs (in progress) |
| `results/invalid_pre_fix/` | - | Archived, invalid: predates an optimizer bug fix. Don't use anything in here. |

## Limitations

- Only one PDE so far (viscous Burgers'); a heat-equation cross-check is in progress (1/12 configs done as of this writing)
- Extended-training comparisons have at most two seeds; one corner is still single-seed
- No configuration we tested actually reaches expressivity sufficiency in either the smooth or shock region of the solution - read the sufficiency numbers as relative distance, not a pass/fail line
- Simulator only, nothing here has touched real quantum hardware
  
## Future Work

- Formalize the circuit-reuse pattern used by `heat_equation.py` into a documented, general PDE interface
- Finish the redundancy/effective-rank-over-epochs sweep (in progress)
- Compare entanglement patterns and measurement operators, expectation values vs. probability vector (in progress)
- Extend Shapley attribution to gate-level and beyond the single q5,r5 config

### Acknowledgements
This project was carried out under [**Qinetic Labs**](https://www.qinetic.org/), with Jithesh Mithra as head researcher, leading the project and responsible for the majority of the implementation, experiments, and analysis. Thanks to Isaac Leon for contributions on the theory side, including work on the Fourier-ceiling framing and the disentanglement analysis.

**Contact**:
- _Email_: jitheshmithra412 [at] gmail [dot] com **and/or** isaacleon0907 [at] gmail [dot] com
- _LinkedIn_: https://www.linkedin.com/in/jitheshmithra/
