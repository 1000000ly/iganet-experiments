"""
E5 — Generalization study: E1 vs E2 across σ_test and N_train.

Architecture (both models): 50 → 512 → 512 → 512 → 100
Training: 30 000 steps, Adam, 6-segment LR decay (1e-3 × 0.5^seg),
          batch = 1 (single random sample per step — matches C++ E1/E2).

E1 label: B-spline coefficients (u)
E2 label: Greville field values  (u_val)

Prerequisites:
  build-tutorials/docs/dataset.pt         (existing, from generate_data)
  build-tutorials/docs/e5_test_s*.pt      (from e5_generate_testsets binary)

Output:
  docs/python/results.csv
"""

import sys, os, time, csv, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_PATH   = os.path.join(os.path.dirname(__file__),
                              '../../build-tutorials/docs/dataset.pt')
TESTSET_DIR    = os.path.join(os.path.dirname(__file__),
                              '../../build-tutorials/docs')
RESULTS_CSV    = os.path.join(os.path.dirname(__file__), 'results.csv')

N_STEPS   = 30_000          # gradient steps (= epochs in C++, batch=1)
N_SEG     = 6
INIT_LR   = 1e-3
SIGMAS    = [0.10, 0.15, 0.20, 0.25]
N_TRAINS  = [50, 100, 200, 400]

SANITY_THRESHOLD = 0.07     # field-L2 must be below 7% for N=400, σ=0.10


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_main():
    """Load dataset.pt → (G, u, u_val, u_grid) all float32."""
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    return (sd['0'].float(),   # G        [500, 50]
            sd['1'].float(),   # u_coeff  [500, 100]
            sd['2'].float(),   # u_val    [500, 100]
            sd['6'].float())   # u_grid   [500, 64, 64]


def load_testset(sigma):
    """Load e5_test_s{tag}.pt → (G, u, u_val, u_grid) float32."""
    tag  = f's{round(sigma*100):03d}'
    path = os.path.join(TESTSET_DIR, f'e5_test_{tag}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Test set not found: {path}\n'
                                f'Run e5_generate_testsets first.')
    m  = torch.jit.load(path, map_location='cpu')
    sd = m.state_dict()
    return (sd['0'].float(),   # G
            sd['1'].float(),   # u
            sd['2'].float(),   # u_val
            sd['4'].float())   # u_grid


def compute_basis_matrices(u_coeff, u_val, u_grid):
    """
    Extract geometry-independent basis matrices from training data.

    B_eval : [100, 4096]  s.t.  u_coeff @ B_eval ≈ u_grid_flat
    M      : [100, 100]   s.t.  u_coeff @ M      ≈ u_val
    M_inv  : [100, 100]   inverse of M (Greville interpolation)
    """
    N = u_coeff.shape[0]
    uc = u_coeff.double()          # [N, 100]
    ug = u_grid.reshape(N, -1).double()   # [N, 4096]
    uv = u_val.double()            # [N, 100]

    pinv_uc = torch.linalg.pinv(uc)        # [100, N]
    B_eval  = (pinv_uc @ ug).float()       # [100, 4096]
    M       = (pinv_uc @ uv).float()       # [100, 100]
    M_inv   = torch.linalg.inv(M.double()).float()  # [100, 100]
    return B_eval, M, M_inv


# ── Network ───────────────────────────────────────────────────────────────────

class SurrogateNet(nn.Module):
    """50 → 512 → 512 → 512 → 100  (identical to C++ E1/E2 architecture)."""
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

def train(G_train, labels_train, device):
    """
    Train SurrogateNet with the same schedule as C++ E1/E2:
    30k random-sample steps, 6-segment Adam LR decay.
    """
    model = SurrogateNet().to(device)
    G     = G_train.to(device)
    Y     = labels_train.to(device)
    N     = G.shape[0]

    steps_per_seg = N_STEPS // N_SEG
    lr = INIT_LR

    for seg in range(N_SEG):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps_per_seg):
            idx = torch.randint(N, (1,)).item()
            loss = F.mse_loss(model(G[idx:idx+1]), Y[idx:idx+1])
            opt.zero_grad()
            loss.backward()
            opt.step()
        lr *= 0.5

    return model.cpu()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, model_type, G_test, u_test, u_val_test, u_grid_test,
             B_eval, M_inv):
    """
    Returns dict with field_l2, coeff_l2, bc_viol (all means over test set).

    model_type : 'E1' (predicts coefficients) or 'E2' (predicts Greville values)
    B_eval     : [100, 4096]   coeff → grid
    M_inv      : [100, 100]    Greville values → coeff (back-projection)
    """
    model.eval()
    N    = G_test.shape[0]
    Ngrid = 64

    sum_fl, sum_cl, max_bc = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i in range(N):
            g       = G_test[i:i+1]       # [1, 50]
            pred    = model(g).squeeze(0)  # [100]

            # Coefficients
            if model_type == 'E1':
                pred_coeff = pred
            else:  # E2: back-project
                pred_coeff = M_inv @ pred

            # Grid field
            pred_grid = (pred_coeff @ B_eval).reshape(Ngrid, Ngrid)  # [64,64]
            true_grid = u_grid_test[i]

            ef = (pred_grid - true_grid).norm().item()
            rf = true_grid.norm().item()
            sum_fl += ef / rf if rf > 0 else ef

            ec = (pred_coeff - u_test[i]).norm().item()
            rc = u_test[i].norm().item()
            sum_cl += ec / rc if rc > 0 else ec

            bc = max(pred_grid[0].abs().max().item(),
                     pred_grid[-1].abs().max().item(),
                     pred_grid[:, 0].abs().max().item(),
                     pred_grid[:, -1].abs().max().item())
            max_bc = max(max_bc, bc)

    model.train()
    return dict(field_l2=sum_fl/N, coeff_l2=sum_cl/N, bc_viol=max_bc)


def trivial_baseline(u_grid_train, u_grid_test):
    mean = u_grid_train.mean(0)
    total = 0.0
    for i in range(u_grid_test.shape[0]):
        ef = (mean - u_grid_test[i]).norm().item()
        rf = u_grid_test[i].norm().item()
        total += ef / rf if rf > 0 else ef
    return total / u_grid_test.shape[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load training pool
    G_all, u_all, uv_all, ug_all = load_main()
    N_pool = 400   # training pool = first 400 samples

    # Precompute basis matrices from the full training pool
    B_eval, M, M_inv = compute_basis_matrices(
        u_all[:N_pool], uv_all[:N_pool], ug_all[:N_pool])
    print(f'B_eval: {list(B_eval.shape)}  M: {list(M.shape)}')

    # Load test sets
    test_sets = {}
    for sigma in SIGMAS:
        try:
            test_sets[sigma] = load_testset(sigma)
            print(f'Loaded test set σ={sigma}: {test_sets[sigma][0].shape[0]} samples')
        except FileNotFoundError as e:
            print(f'ERROR: {e}')
            sys.exit(1)

    # ── Sanity check ──────────────────────────────────────────────────────────
    print('\n── Sanity check: E1 & E2 at N=400, σ=0.10 ──')
    G_tr400 = G_all[:400]
    Gt, ut, uvt, ugt = test_sets[0.10]

    for mtype, label_tr in [('E1', u_all[:400]), ('E2', uv_all[:400])]:
        t0 = time.time()
        m  = train(G_tr400, label_tr, device)
        r  = evaluate(m, mtype, Gt, ut, uvt, ugt, B_eval, M_inv)
        dt = time.time() - t0
        print(f'  {mtype}: field_l2={r["field_l2"]*100:.2f}%  '
              f'coeff_l2={r["coeff_l2"]*100:.2f}%  t={dt:.0f}s')
        if r['field_l2'] > SANITY_THRESHOLD:
            print(f'  FAIL: field_l2 > {SANITY_THRESHOLD*100:.0f}%. '
                  f'Re-implementation mismatch. Stopping.')
            sys.exit(1)
    print('  Sanity check PASSED.\n')

    # ── Full sweep ────────────────────────────────────────────────────────────
    rows = []
    header = ['model', 'N_train', 'sigma_test',
              'field_l2', 'coeff_l2', 'bc_viol', 'trivial_l2']

    total = len(['E1','E2']) * len(N_TRAINS) * len(SIGMAS)
    done  = 0

    for N_tr in N_TRAINS:
        G_tr = G_all[:N_tr]
        u_tr = u_all[:N_tr]
        uv_tr = uv_all[:N_tr]
        ug_tr = ug_all[:N_tr]

        print(f'\n── N_train={N_tr} ──')
        for mtype, label_tr in [('E1', u_tr), ('E2', uv_tr)]:
            t0 = time.time()
            model = train(G_tr, label_tr, device)
            dt    = time.time() - t0
            print(f'  {mtype} trained in {dt:.0f}s', end='')

            for sigma in SIGMAS:
                Gt, ut, uvt, ugt = test_sets[sigma]
                triv = trivial_baseline(ug_tr, ugt)
                r    = evaluate(model, mtype, Gt, ut, uvt, ugt, B_eval, M_inv)

                row = [mtype, N_tr, sigma,
                       round(r['field_l2'],  6),
                       round(r['coeff_l2'],  6),
                       round(r['bc_viol'],   8),
                       round(triv,           6)]
                rows.append(row)
                done += 1
                print(f'  σ={sigma:.2f} fl={r["field_l2"]*100:.2f}%', end='')

            print()  # newline after all sigmas

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f'\nResults saved to {RESULTS_CSV}  ({len(rows)} rows)')

    # Quick summary table
    print(f'\n{"model":>5} {"N":>5} {"σ_test":>7} {"field_l2":>9} {"coeff_l2":>9}')
    print('-' * 45)
    for r in rows:
        print(f'{r[0]:>5} {r[1]:>5} {r[2]:>7.2f} '
              f'{r[3]*100:>8.2f}% {r[4]*100:>8.2f}%')


if __name__ == '__main__':
    main()
