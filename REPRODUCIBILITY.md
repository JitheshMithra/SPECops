# Reproducibility

## Environment

Tested with:

- Python 3.14.5 (anything >=3.10 should work)
- pennylane 0.45.1
- numpy 2.5.1
- scipy 1.18.0
- matplotlib 3.11.1
- psutil 7.2.2

No `torch` dependency anywhere in this repo — `frequency_unit_conversion.py` deliberately uses PennyLane's own autodiff (`qp.jacobian`/`qp.grad`) instead, since that's what the rest of the codebase (`pde_residual()`) is already built on.

Install with:

```
pip install pennylane numpy scipy matplotlib psutil
```
All scripts assume they're run from `SPECops/src/` (they resolve `results/` and `sweep_checkpoints/` relative to their own file location, not the CWD, so this isn't strictly required, but paths below assume it).

## Known issue affecting anything trained before 2026-08-03

`loop.py`'s training step originally called `opt.step(cost_fn, params)`, passing all five weight arrays bundled as one Python tuple. PennyLane's autograd only tracks `requires_grad` on individual arrays, not on a container, so the gradient for that bundled argument was always empty and `opt.step` silently returned the input unchanged — every "trained" run was actually just a random init, evaluated. This affected both `main.py` (quantum) and `main_classical.py` (shared training loop), which is why an early classical run reported `l2_error ≈ 0.97` (indistinguishable from an untrained network). Fixed by unpacking params as separate positional args: `opt.step(cost_fn, *params)`. Checkpoints/logs produced before the fix are kept for reference under `results/invalid_pre_fix/` and `sweep_checkpoints/invalid_pre_fix/` — don't reuse them.

## Running each script from a clean checkout

| Script | Command | Needs a checkpoint? | Output |
|---|---|---|---|
| `main.py` | `python main.py` | no | smoke test only (prints one forward pass + loss), no files written |
| `main_classical.py` | `python main_classical.py` | no | same, classical control architecture |
| `loop.py` | not usually run directly — `train()` is called by `sweep.py`/`run_classical_comparison.py` | no | saves a checkpoint if `checkpointPath` is passed |
| `eval.py` | `python eval.py --checkpoint <path.pkl> --model main` (or `--model main_classical`) | yes | prints relative L2 error vs. the Cole-Hopf reference and PDE residual error; appends a row to `results/eval_results.csv` |
| `sweep.py` | `python sweep.py` (full grid: 3 qubits × 4 reuploads × 3 seeds = 36 runs) | no (trains fresh) | one row per `(n_qubits, n_reuploads, seed)` appended to `results/sweep_results.csv` as each seed finishes; checkpoints saved to `sweep_checkpoints/q{n}_r{n}_s{seed}.pkl`. Safe to interrupt — rerunning skips any `(n_qubits, n_reuploads, seed)` already present in the CSV. `python sweep.py --full-grid --n-qubits N --n-reuploads N --seed S` re-evaluates one existing checkpoint at full 256×100 resolution into `results/sweep_full_grid_results.csv` |
| `run_classical_comparison.py` | `python run_classical_comparison.py` (3 seeds) | no (trains fresh) | one aggregated row in `results/classical_comparison.csv`; checkpoints to `sweep_checkpoints/classical_s{seed}.pkl` |
| `gradient_variance.py` | `python gradient_variance.py` (full grid, each config in its own subprocess for memory isolation) | no (random inits only) | `results/gradient_variance_results.csv`: mean gradient variance of the quantum layer's weights per `(n_qubits, n_reuploads)` — a barren-plateau signature, unaffected by the training bug above |
| `fourier_spectrum.py` | `python fourier_spectrum.py` | no (structural property of the circuit) | `results/fourier_spectrum_results.csv`: each config's accessible Fourier frequency spectrum |
| `shock_spectrum.py` | `python shock_spectrum.py` | no (analytic reference only) | `results/burgers_shock_spectrum.json`: FFT of the Cole-Hopf reference at 5 time snapshots, with the 99%-energy effective bandwidth per snapshot |
| `frequency_unit_conversion.py` | `python frequency_unit_conversion.py` — **run after `fourier_spectrum.py` and `shock_spectrum.py`**, it reads both of their outputs | no (random inits, only needs `d(theta)/dx` of the pre-layer) | `results/frequency_unit_conversion.csv` (converts each config's circuit frequency ceiling from encoding-angle units to physical-x units), `results/expressivity_vs_shock.csv` (sufficiency table: does the config's frequency ceiling cover the shock's required bandwidth at each snapshot), `results/pre_layer_slope_vs_x.png` |
| `activation_analysis.py` | `python activation_analysis.py --classical-checkpoint <path.pkl> --quantum-checkpoint <path.pkl>` | yes, both | `results/activation_analysis.csv` (mean/std activation per neuron, pre-layer/post-quantum-layer/output, for both models) and `results/activation_comparison.png` |

## Output file glossary

- `sweep_results.csv` — per-seed accuracy/timing rows for the quantum sweep. Columns: `n_qubits, n_reuploads, seed, l2_error, pde_residual_error, final_train_loss, train_time_sec, param_count, peak_memory_mb`. Aggregate mean/std per config from these rows directly (grouped by `n_qubits, n_reuploads`) — the script no longer writes a pre-aggregated row.
- `classical_comparison.csv` — same metrics as above but pre-aggregated (mean/std across seeds) for the non-quantum control architecture, for a direct comparison against any sweep config.
- `gradient_variance_results.csv` — barren-plateau check: variance of `d(pde_loss)/d(W_q)` across 50 random inits per config, on a fixed collocation batch. Decreasing toward zero as `n_qubits` grows is the barren-plateau signature.
- `fourier_spectrum_results.csv` — the set of integer frequencies each config's circuit can represent, in units of the encoding angle (post pre-layer tanh), not physical `x`.
- `burgers_shock_spectrum.json` — the actual spatial frequency content of the Burgers' shock (from the exact Cole-Hopf solution), in physical `x` units, at `t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`.
- `frequency_unit_conversion.csv` / `expressivity_vs_shock.csv` — bridges the two units above via the pre-layer's local slope, then checks whether each config's frequency ceiling (in physical-x units) is sufficient to represent the shock's required bandwidth at each snapshot.
- `activation_analysis.csv` — mean/std of pre-layer, post-quantum-layer, and final output activations for a trained classical vs. quantum checkpoint, evaluated on the same `(t,x)` grid.
