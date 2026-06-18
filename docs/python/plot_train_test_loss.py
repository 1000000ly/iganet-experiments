"""
Train-loss vs test-loss curves for E1 and E2 on the same figure.

Setup:
  N_train = 400 (first 400 of dataset.pt), N_test = 100 (last 100)
  30 000 steps, 6-segment Adam LR decay — identical to E1/E2 C++ spec.
  Single seed (42) for reproducibility.

Every EVAL_EVERY steps:
  - Compute full-set train MSE  (avg over all 400 training samples)
  - Compute full-set test  MSE  (avg over all 100 test samples)

Both losses are normalised to 1 at step 0 → directly comparable.
Y-axis is log scale so the multi-order-magnitude decay is visible.

Output: docs/python/train_test_loss.png
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, '../../build-tutorials/docs/dataset.pt')
OUT_PNG      = os.path.join(HERE, 'train_test_loss.png')

N_STEPS    = 30_000
N_SEG      = 6
INIT_LR    = 1e-3
EVAL_EVERY = 300        # evaluate both sets every N steps (100 checkpoints)
SEED       = 42
N_TR       = 400
N_TE       = 100

COLORS = {'E1': '#2166ac', 'E2': '#d6604d'}


# ── Data ──────────────────────────────────────────────────────────────────────

def load():
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    G   = sd['0'].float()     # [500, 50]
    u   = sd['1'].float()     # [500, 100]   B-spline coefficients
    uv  = sd['2'].float()     # [500, 100]   Greville field values
    return G[:N_TR], u[:N_TR], uv[:N_TR], G[N_TR:], u[N_TR:], uv[N_TR:]


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


# ── MSE helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def full_mse(model, G, Y):
    """Average MSE over the entire set (no batching needed for 400 samples)."""
    pred = model(G)                              # [N, 100]
    return F.mse_loss(pred, Y).item()


# ── Training loop with checkpointing ─────────────────────────────────────────

def train_and_record(G_tr, Y_tr, G_te, Y_te, label, device):
    torch.manual_seed(SEED)
    model = SurrogateNet().to(device)

    G_tr_d  = G_tr.to(device)
    Y_tr_d  = Y_tr.to(device)
    G_te_d  = G_te.to(device)
    Y_te_d  = Y_te.to(device)

    steps_per_seg = N_STEPS // N_SEG
    lr = INIT_LR

    steps_log, train_log, test_log = [], [], []
    step = 0

    def record():
        model.eval()
        tr_mse = full_mse(model, G_tr_d, Y_tr_d)
        te_mse = full_mse(model, G_te_d, Y_te_d)
        model.train()
        steps_log.append(step)
        train_log.append(tr_mse)
        test_log.append(te_mse)

    record()   # step 0 — used for normalisation

    t0 = time.time()
    for seg in range(N_SEG):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps_per_seg):
            idx  = torch.randint(G_tr_d.shape[0], (1,)).item()
            loss = F.mse_loss(model(G_tr_d[idx:idx+1]), Y_tr_d[idx:idx+1])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                record()
        lr *= 0.5

    print(f'  {label}: done in {time.time()-t0:.0f}s  '
          f'final train={train_log[-1]:.3e}  test={test_log[-1]:.3e}')

    # Normalise to step-0 value
    norm = train_log[0]
    train_norm = [v / norm for v in train_log]
    test_norm  = [v / norm for v in test_log]

    return np.array(steps_log), np.array(train_norm), np.array(test_norm)


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(results):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    lr_boundaries = [N_STEPS * k // N_SEG for k in range(1, N_SEG)]

    for col in lr_boundaries:
        ax.axvline(col, color='gray', linewidth=0.6, linestyle=':', alpha=0.6)

    for label, (steps, train_n, test_n) in results.items():
        c = COLORS[label]
        ax.semilogy(steps, train_n, '-',  color=c, linewidth=1.6,
                    label=f'{label} train')
        ax.semilogy(steps, test_n,  '--', color=c, linewidth=1.6,
                    alpha=0.85, label=f'{label} test')

    # Annotate LR decay points
    for i, x in enumerate(lr_boundaries):
        ax.text(x + 200, ax.get_ylim()[1] * 0.92,
                f'LR/2', fontsize=6.5, color='gray',
                rotation=90, va='top')

    ax.set_xlabel('Training step', fontsize=11)
    ax.set_ylabel('Normalised MSE loss  (loss / loss₀)', fontsize=11)
    ax.set_title('E1 vs E2 — train & test loss curves\n'
                 f'(N_train={N_TR}, N_test={N_TE}, seed={SEED})', fontsize=10)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, N_STEPS)
    ax.grid(True, which='both', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nPlot saved: {OUT_PNG}')
    assert os.path.exists(OUT_PNG)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Recording every {EVAL_EVERY} steps → '
          f'{N_STEPS // EVAL_EVERY} checkpoints per model\n')

    G_tr, u_tr, uv_tr, G_te, u_te, uv_te = load()

    results = {}

    print('── E1 (label = B-spline coefficients) ──')
    results['E1'] = train_and_record(G_tr, u_tr,  G_te, u_te,  'E1', device)

    print('── E2 (label = Greville field values) ──')
    results['E2'] = train_and_record(G_tr, uv_tr, G_te, uv_te, 'E2', device)

    make_plot(results)

    # Summary
    print('\nFinal normalised loss (lower = trained more):')
    for lbl, (_, tr, te) in results.items():
        print(f'  {lbl}: train={tr[-1]:.4f}  test={te[-1]:.4f}  '
              f'gap={abs(te[-1]-tr[-1]):.4f}')


if __name__ == '__main__':
    main()
