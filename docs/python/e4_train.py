"""
E4 — FNO2d Training
Geometry -> field solution on 64x64 parametric grid.

Architecture:  FNO2d(modes=12, width=16, n_layers=4)  ~593k params
Training:      6 segments × 5000 epochs, LR halved per segment (same schedule as E1)
               Batch size 20, Adam optimizer
Device:        MPS (float32) if available, else CPU (float32)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from fno2d import FNO2d

DATASET_PATH = os.path.join(os.path.dirname(__file__),
                            '../../build-tutorials/docs/dataset.pt')
MODEL_OUT     = os.path.join(os.path.dirname(__file__), 'e4_fno.pt')

MODES          = 12
WIDTH          = 16
N_LAYERS       = 4
BATCH_SIZE     = 20
N_SEGMENTS     = 6
EPOCHS_PER_SEG = 5000
INIT_LR        = 1e-3


def load_dataset():
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    G_grid = sd['5'].float()   # [500, 64, 64, 2]
    u_grid = sd['6'].float()   # [500, 64, 64]
    return G_grid, u_grid


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[E4] Device: {device}')

    G_grid, u_grid = load_dataset()
    N       = G_grid.shape[0]
    N_train = int(N * 0.8)
    N_test  = N - N_train

    G_train = G_grid[:N_train]
    u_train = u_grid[:N_train]

    print(f'[E4] G_grid shape: {list(G_grid.shape)}, u_grid shape: {list(u_grid.shape)}')
    print(f'[E4] N={N}, train={N_train}, test={N_test}')
    print(f'[E4] u_grid range: [{u_grid.min().item():.4f}, {u_grid.max().item():.4f}]')

    dataset = TensorDataset(G_train, u_train)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ── Sanity check: 1000 epochs ──────────────────────────────────────────────
    print('\n[E4] Sanity check (1000 epochs) ...')
    model_sc = FNO2d(MODES, MODES, WIDTH, N_LAYERS).to(device)
    n_params  = sum(p.numel() for p in model_sc.parameters() if p.requires_grad)
    print(f'[E4] Parameters: {n_params:,}')

    opt = torch.optim.Adam(model_sc.parameters(), lr=INIT_LR)
    t0  = time.time()
    last_loss = float('inf')
    for ep in range(1000):
        ep_loss = 0.0
        for G_b, u_b in loader:
            G_b, u_b = G_b.to(device), u_b.to(device)
            opt.zero_grad()
            u_pred = model_sc(G_b)
            loss   = F.mse_loss(u_pred, u_b)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        last_loss = ep_loss / len(loader)
    print(f'[E4] Sanity check done: loss={last_loss:.6e}  time={time.time()-t0:.1f}s')

    if last_loss > 1.0:
        print('[E4] WARNING: loss did not decrease below 1.0 — check architecture')
        return

    print('[E4] Sanity check passed. Starting full training (30 000 epochs)...\n')

    # ── Full training ──────────────────────────────────────────────────────────
    model = FNO2d(MODES, MODES, WIDTH, N_LAYERS).to(device)
    lr    = INIT_LR

    for seg in range(N_SEGMENTS):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        t0  = time.time()
        ep_losses = []

        for ep in range(EPOCHS_PER_SEG):
            ep_loss = 0.0
            for G_b, u_b in loader:
                G_b, u_b = G_b.to(device), u_b.to(device)
                opt.zero_grad()
                u_pred = model(G_b)
                loss   = F.mse_loss(u_pred, u_b)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            ep_losses.append(ep_loss / len(loader))

        elapsed = time.time() - t0
        print(f'[E4] Segment {seg+1}/{N_SEGMENTS}  lr={lr:.2e}  '
              f'loss={ep_losses[-1]:.6e}  time={elapsed:.1f}s')
        lr *= 0.5

    # Save on CPU
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), MODEL_OUT)
    print(f'\n[E4] Model saved to {MODEL_OUT}')


if __name__ == '__main__':
    main()
