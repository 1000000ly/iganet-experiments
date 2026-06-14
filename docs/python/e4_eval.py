"""
E4 — FNO2d Evaluation
Loads saved model, computes 4 metrics on test set, prints comparison table.

Metrics:
  1. Field-space relative L2    (E1 baseline: 3.09%)
  2. Boundary violation          (E1 baseline: 1.51e-4)
  3. Parameter count             (E1 baseline: ~580k)
  4. Inference time on CPU       (E1 baseline: 0.062ms)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fno2d import FNO2d

DATASET_PATH = os.path.join(os.path.dirname(__file__),
                            '../../build-tutorials/docs/dataset.pt')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'e4_fno.pt')

MODES    = 12
WIDTH    = 16
N_LAYERS = 4
TIMING_REPS = 1000


def load_dataset():
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    G_grid = sd['5'].float()   # [500, 64, 64, 2]
    u_grid = sd['6'].float()   # [500, 64, 64]
    return G_grid, u_grid


def main():
    G_grid, u_grid = load_dataset()
    N       = G_grid.shape[0]
    N_train = int(N * 0.8)
    N_test  = N - N_train

    G_test  = G_grid[N_train:]   # [100, 64, 64, 2]
    u_test  = u_grid[N_train:]   # [100, 64, 64]
    u_train = u_grid[:N_train]   # [400, 64, 64]

    model = FNO2d(MODES, MODES, WIDTH, N_LAYERS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Field-space L2 and boundary violation ──────────────────────────────────
    sum_field_l2 = 0.0
    max_bc_viol  = 0.0

    with torch.no_grad():
        for i in range(N_test):
            g      = G_test[i:i+1]           # [1, 64, 64, 2]
            u_pred = model(g).squeeze(0)      # [64, 64]
            u_ref  = u_test[i]                # [64, 64]

            ef = (u_pred - u_ref).norm().item()
            rf = u_ref.norm().item()
            sum_field_l2 += ef / rf if rf > 0 else ef

            bc = max(
                u_pred[0].abs().max().item(),     # row 0  (xi2=0)
                u_pred[-1].abs().max().item(),    # row 63 (xi2=1)
                u_pred[:, 0].abs().max().item(),  # col 0  (xi1=0)
                u_pred[:, -1].abs().max().item(), # col 63 (xi1=1)
            )
            max_bc_viol = max(max_bc_viol, bc)

    mean_field_l2 = sum_field_l2 / N_test

    # ── Trivial baseline (predict mean training field for all test samples) ─────
    u_mean = u_train.mean(0)  # [64, 64]
    trivial_l2 = 0.0
    for i in range(N_test):
        ef = (u_mean - u_test[i]).norm().item()
        rf = u_test[i].norm().item()
        trivial_l2 += ef / rf if rf > 0 else ef
    trivial_l2 /= N_test

    # ── Inference timing on CPU ───────────────────────────────────────────────
    g_single = G_test[0:1]

    with torch.no_grad():
        for _ in range(200):        # warmup
            _ = model(g_single)

        t0 = time.perf_counter()
        for _ in range(TIMING_REPS):
            _ = model(g_single)
        infer_ms = (time.perf_counter() - t0) / TIMING_REPS * 1000.0

    # ── Print table ───────────────────────────────────────────────────────────
    print(f'\n{"Metric":<24} {"E1 (IgANets)":<18} {"E4 (FNO)"}')
    print('-' * 62)
    print(f'{"Field-space L2":<24} {"3.09%":<18} {mean_field_l2*100:.2f}%')
    print(f'{"Boundary violation":<24} {"1.51e-4":<18} {max_bc_viol:.2e}')
    print(f'{"Parameter count":<24} {"~580k":<18} {n_params/1000:.0f}k')
    print(f'{"Inference time (CPU)":<24} {"0.062ms":<18} {infer_ms:.3f}ms')
    print(f'{"Trivial baseline":<24} {"11.7%":<18} {trivial_l2*100:.2f}%')
    print()
    print(f'[E4] n_params exact: {n_params:,}')
    print(f'[E4] N_test={N_test}  modes={MODES}  width={WIDTH}  n_layers={N_LAYERS}')


if __name__ == '__main__':
    main()
