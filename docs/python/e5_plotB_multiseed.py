"""
E5 Plot B (multi-seed): data-efficiency curve with error bars.

Fixed test set: e5_test_s010.pt (σ=0.10 interpolation set).
N_train ∈ {50, 100, 200, 400}, 5 seeds each, models E1 and E2.
Each seed draws a fresh random N_train-sized subset of the 400-sample
training pool and uses the same seed for weight initialisation.

Outputs (written to docs/python/):
  plotB_runs.csv   — one row per (model, N_train, seed)
  plotB.png        — mean ± 1 std error bars, both models
"""

import sys, os, time, csv, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, '../../build-tutorials/docs/dataset.pt')
TESTSET_PATH = os.path.join(HERE, '../../build-tutorials/docs/e5_test_s010.pt')
RUNS_CSV     = os.path.join(HERE, 'plotB_runs.csv')
PLOT_PNG     = os.path.join(HERE, 'plotB.png')

# ── Hyperparameters (unchanged from E1/E2 C++ spec) ──────────────────────────
N_STEPS   = 30_000
N_SEG     = 6
INIT_LR   = 1e-3
SEEDS     = [0, 1, 2, 3, 4]
N_TRAINS  = [50, 100, 200, 400]
SANITY_N  = 400          # use N=400 for sanity check
SANITY_MAX = 0.07        # mean field-L2 at N=400 must be below 7%

COLORS = {'E1': '#2166ac', 'E2': '#d6604d'}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_main():
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    return (sd['0'].float(),   # G        [500, 50]
            sd['1'].float(),   # u_coeff  [500, 100]
            sd['2'].float(),   # u_val    [500, 100]
            sd['6'].float())   # u_grid   [500, 64, 64]


def load_testset():
    m  = torch.jit.load(TESTSET_PATH, map_location='cpu')
    sd = m.state_dict()
    return (sd['0'].float(),   # G
            sd['1'].float(),   # u_coeff
            sd['2'].float(),   # u_val
            sd['4'].float())   # u_grid


# ── Basis matrices (ridge regression, geometry-independent) ───────────────────

def compute_basis_matrices(u_coeff, u_val, u_grid):
    N  = u_coeff.shape[0]
    uc = u_coeff.double()
    ug = u_grid.reshape(N, -1).double()
    uv = u_val.double()

    A   = uc.T @ uc
    lam = 1e-4 * A.diagonal().mean().item()
    reg = lam * torch.eye(100, dtype=torch.float64)
    AtA = A + reg

    B_eval = torch.linalg.solve(AtA, uc.T @ ug).float()   # [100, 4096]
    M      = torch.linalg.solve(AtA, uc.T @ uv).float()   # [100, 100]
    M_inv  = torch.linalg.pinv(M.double()).float()
    return B_eval, M_inv


# ── Network ───────────────────────────────────────────────────────────────────

class SurrogateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 512),  nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 100),
        )
    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(G_train, labels_train, seed, device):
    torch.manual_seed(seed)
    model = SurrogateNet().to(device)
    G = G_train.to(device)
    Y = labels_train.to(device)
    N = G.shape[0]

    steps_per_seg = N_STEPS // N_SEG
    lr = INIT_LR
    for _ in range(N_SEG):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps_per_seg):
            idx  = torch.randint(N, (1,)).item()
            loss = F.mse_loss(model(G[idx:idx+1]), Y[idx:idx+1])
            opt.zero_grad(); loss.backward(); opt.step()
        lr *= 0.5

    return model.cpu()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, model_type, G_test, u_test, u_grid_test, B_eval, M_inv):
    model.eval()
    N = G_test.shape[0]
    sum_fl = 0.0

    with torch.no_grad():
        for i in range(N):
            pred = model(G_test[i:i+1]).squeeze(0)          # [100]
            pred_coeff = pred if model_type == 'E1' else pred @ M_inv
            pred_grid  = (pred_coeff @ B_eval).reshape(64, 64)
            true_grid  = u_grid_test[i]
            ef = (pred_grid - true_grid).norm().item()
            rf = true_grid.norm().item()
            sum_fl += ef / rf if rf > 0 else ef

    model.train()
    return sum_fl / N


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_subset(G_pool, label_pool, N, seed):
    """Draw a fresh random N-sample subset from the pool (reproducible)."""
    rng   = torch.Generator().manual_seed(seed * 997 + N * 31)  # decorrelated
    perm  = torch.randperm(G_pool.shape[0], generator=rng)[:N]
    return G_pool[perm], label_pool[perm]


# ── Single-run cost check ─────────────────────────────────────────────────────

def cost_check(G_pool, u_pool, uv_pool, G_test, u_test, ug_test,
               B_eval, M_inv, device):
    print('\n── Cost check: one E1 training at N_train=400, seed=0 ──')
    Gtr, Ytr = sample_subset(G_pool, u_pool, 400, seed=0)
    t0 = time.time()
    m  = train_one(Gtr, Ytr, seed=0, device=device)
    dt = time.time() - t0
    fl = evaluate(m, 'E1', G_test, u_test, ug_test, B_eval, M_inv)
    total_est = dt * len(SEEDS) * len(N_TRAINS) * 2
    print(f'  Single training: {dt:.1f}s   field-L2: {fl*100:.2f}%')
    print(f'  Estimated total (40 runs): {total_est/60:.1f} min')
    return dt, fl


# ── Full sweep ────────────────────────────────────────────────────────────────

def run_sweep(G_pool, u_pool, uv_pool, G_test, u_test, ug_test,
              B_eval, M_inv, device):
    rows   = []
    header = ['model', 'N_train', 'seed', 'field_l2']
    total  = len(SEEDS) * len(N_TRAINS) * 2
    done   = 0
    t_sweep = time.time()

    for N_tr in N_TRAINS:
        print(f'\n── N_train={N_tr} ──')
        for mtype, label_pool in [('E1', u_pool), ('E2', uv_pool)]:
            fls = []
            for seed in SEEDS:
                Gtr, Ytr = sample_subset(G_pool, label_pool, N_tr, seed)
                t0  = time.time()
                mdl = train_one(Gtr, Ytr, seed=seed, device=device)
                dt  = time.time() - t0
                fl  = evaluate(mdl, mtype, G_test, u_test, ug_test, B_eval, M_inv)
                fls.append(fl)
                rows.append([mtype, N_tr, seed, round(fl, 6)])
                done += 1
                print(f'  {mtype} N={N_tr} seed={seed}: '
                      f'fl={fl*100:.2f}%  ({dt:.0f}s)  [{done}/{total}]')

            mu  = float(np.mean(fls))
            std = float(np.std(fls, ddof=1))
            print(f'  → {mtype} N={N_tr}: mean={mu*100:.2f}%  std={std*100:.2f}%')

    print(f'\nSweep done: {(time.time()-t_sweep)/60:.1f} min  ({len(rows)} rows)')
    return rows, header


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(rows):
    # Aggregate
    data = {}
    for r in rows:
        key = (r[0], r[1])    # (model, N_train)
        data.setdefault(key, []).append(r[3])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = sorted(set(r[1] for r in rows))

    for mtype in ('E1', 'E2'):
        means = [np.mean(data[(mtype, n)]) * 100 for n in xs]
        stds  = [np.std(data[(mtype, n)], ddof=1) * 100 for n in xs]
        ax.errorbar(xs, means, yerr=stds, fmt='o-',
                    color=COLORS[mtype], label=mtype,
                    linewidth=1.8, markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2)

    ax.set_xlabel('Training set size N_train', fontsize=11)
    ax.set_ylabel('Field-space relative L2 (%)', fontsize=11)
    ax.set_title('E5 — Data efficiency (5 seeds, σ_test = 0.10)', fontsize=10)
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nPlot saved: {PLOT_PNG}')
    assert os.path.exists(PLOT_PNG), 'Plot was not written!'


# ── Readout ───────────────────────────────────────────────────────────────────

def print_readout(rows):
    data = {}
    for r in rows:
        data.setdefault((r[0], r[1]), []).append(r[3])

    print(f'\n{"N_train":>8}  {"E1 mean±std":>14}  {"E2 mean±std":>14}  Note')
    print('-' * 65)
    for n in sorted(set(r[1] for r in rows)):
        e1 = data[('E1', n)]
        e2 = data[('E2', n)]
        mu1, s1 = np.mean(e1)*100, np.std(e1, ddof=1)*100
        mu2, s2 = np.mean(e2)*100, np.std(e2, ddof=1)*100
        # Distinguishable if the means differ by more than the larger std
        gap     = abs(mu1 - mu2)
        overlap = max(s1, s2)
        note    = 'distinguishable' if gap > overlap else 'overlapping'
        print(f'{n:>8}  {mu1:>6.2f}% ± {s1:>4.2f}%  '
              f'{mu2:>6.2f}% ± {s2:>4.2f}%  {note}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost-check-only', action='store_true',
                    help='Run one training, print time estimate, then exit.')
    args = ap.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    G_all, u_all, uv_all, ug_all = load_main()
    G_pool  = G_all[:400]
    u_pool  = u_all[:400]
    uv_pool = uv_all[:400]

    G_test, u_test, uv_test, ug_test = load_testset()

    ug_pool = ug_all[:400]
    B_eval, M_inv = compute_basis_matrices(u_pool, uv_pool, ug_pool)

    dt, fl_check = cost_check(G_pool, u_pool, uv_pool,
                               G_test, u_test, ug_test,
                               B_eval, M_inv, device)

    if args.cost_check_only:
        print('\n[--cost-check-only] Stopping here. Re-run without the flag to '
              'launch the full 40-run sweep.')
        return

    # ── Full sweep ────────────────────────────────────────────────────────────
    rows, header = run_sweep(G_pool, u_pool, uv_pool,
                             G_test, u_test, ug_test,
                             B_eval, M_inv, device)

    # Sanity: N=400 mean field-L2
    n400 = [r[3] for r in rows if r[1] == 400]
    mean400 = float(np.mean(n400))
    print(f'\nSanity @ N=400: mean field-L2 = {mean400*100:.2f}%')
    if mean400 > SANITY_MAX:
        print(f'WARNING: exceeds sanity threshold {SANITY_MAX*100:.0f}% — '
              f'check training.')

    # Save CSV
    with open(RUNS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'Runs saved: {RUNS_CSV}  ({len(rows)} rows)')

    # Plot
    make_plot(rows)

    # Print readout
    print_readout(rows)


if __name__ == '__main__':
    main()
