# IgANets Special Topic: Poisson Surrogate Modeling

Special topic project for TU Delft Computational Science track.
Authors: Margot van de Klok, Xuan Du (Claire)
Supervisor: Prof. Matthias Möller

## Overview

This project investigates structure-preserving neural surrogate models
for the Poisson equation on parametric geometries, building on the
IgANets framework. Four experiments are conducted:

- **E1**: B-spline coefficient prediction (3.09% relative L2)
- **E2**: Greville field-value prediction baseline (3.19%)
- **E3**: Loss function ablation (residual vs supervised)
- **E4**: Comparison with Fourier Neural Operator

## Reproducing experiments

Requirements: LibTorch, PyTorch (for E4), CMake.

```bash
# Build C++ experiments
mkdir build && cd build
cmake .. && make

# Generate dataset
./docs/generate_data

# Run experiments
./docs/e1_coeff_prediction
./docs/e2_field_prediction
./docs/e3_alpha_sweep
./docs/e3_diagnostics

# Run E4 (FNO baseline)
cd ../docs/python
python e4_train.py
python e4_eval.py
```

## Key results

| Experiment | Metric | Value |
|---|---|---|
| E1 | Field-space L2 | 3.09% |
| E1 | Boundary violation | 1.51e-4 |
| E2 | Field-space L2 | 3.19% |
| E3 | Phase transition at α ≈ 1/κ(A) ≈ 0.001 | - |
| E4 (FNO) | Field-space L2 | 2.26% |
| E4 (FNO) | Inference time | 4.13ms (vs E1's 0.062ms) |

See `report.pdf` for full analysis.

## Acknowledgments

This project uses the IgANets framework by Matthias Möller et al.
The EZSolver was modified to incorporate Jacobian-based metric tensors
for physical-domain Laplacian assembly (see `include/solver/ezsolver.hpp`).

## Repository structure

```
docs/
  generate_data.cxx          # Dataset generation (500 Poisson samples)
  e1_coeff_prediction.cxx    # E1: IgANet coefficient prediction
  e2_field_prediction.cxx    # E2: IgANet Greville field prediction
  e3_alpha_sweep.cxx         # E3: Loss ablation (α sweep)
  e3_diagnostics.cxx         # E3: Fine-grained α and LR diagnostics
  verify_solver.cxx          # EZSolver correctness check
  python/
    fno2d.py                 # FNO2d architecture
    e4_train.py              # E4: FNO training
    e4_eval.py               # E4: FNO evaluation
    e4_final.py              # E4: Combined train+eval
    verify_dataset.py        # Dataset integrity check
    e4_fno_ep1000.pt         # Trained FNO checkpoint (ep=1000)
include/solver/ezsolver.hpp  # Modified: physical-domain Laplacian
```
