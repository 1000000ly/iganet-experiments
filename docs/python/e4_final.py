"""
E4 Final: train to ep=1000, save model, evaluate all metrics.
"""

import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from fno2d import FNO2d

DATASET_PATH = os.path.join(os.path.dirname(__file__),
                            '../../build-tutorials/docs/dataset.pt')
MODEL_OUT    = os.path.join(os.path.dirname(__file__), 'e4_fno_ep1000.pt')

MODES = 12; WIDTH = 16; N_LAYERS = 4; BATCH = 20


def field_l2_set(model, G, u, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(G.shape[0]):
            pred = model(G[i:i+1].to(device)).cpu().squeeze(0)
            ef = (pred - u[i]).norm().item()
            rf = u[i].norm().item()
            total += ef / rf if rf > 0 else ef
    model.train()
    return total / G.shape[0]


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    G_all = sd['5'].float(); u_all = sd['6'].float()
    N = G_all.shape[0]; N_tr = int(N * 0.8)
    G_tr, u_tr = G_all[:N_tr], u_all[:N_tr]
    G_te, u_te = G_all[N_tr:], u_all[N_tr:]

    loader = DataLoader(TensorDataset(G_tr, u_tr), batch_size=BATCH, shuffle=True)

    model = FNO2d(MODES, MODES, WIDTH, N_LAYERS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    for ep in range(1000):
        for G_b, u_b in loader:
            G_b, u_b = G_b.to(device), u_b.to(device)
            opt.zero_grad()
            F.mse_loss(model(G_b), u_b).backward()
            opt.step()
        if (ep + 1) in (30, 100, 300, 500, 1000):
            print(f'ep={ep+1:4d}  t={time.time()-t0:.0f}s', flush=True)

    print(f'Training done: {time.time()-t0:.0f}s', flush=True)

    # Save model (CPU)
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), MODEL_OUT)
    print(f'Saved: {MODEL_OUT}', flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model_cpu.eval()

    n_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)

    # Field L2 (test)
    sum_l2 = 0.0
    max_bc  = 0.0
    with torch.no_grad():
        for i in range(len(G_te)):
            pred = model_cpu(G_te[i:i+1]).squeeze(0)   # [64, 64]
            ref  = u_te[i]
            ef = (pred - ref).norm().item()
            rf = ref.norm().item()
            sum_l2 += ef / rf if rf > 0 else ef
            bc = max(pred[0].abs().max().item(),
                     pred[-1].abs().max().item(),
                     pred[:, 0].abs().max().item(),
                     pred[:, -1].abs().max().item())
            max_bc = max(max_bc, bc)
    mean_l2 = sum_l2 / len(G_te)

    # Trivial baseline
    u_mean = u_tr.mean(0)
    trivial = sum((u_mean - u_te[i]).norm().item() /
                  (u_te[i].norm().item() or 1.0)
                  for i in range(len(u_te))) / len(u_te)

    # Inference time (CPU, single sample, 1000 runs)
    g0 = G_te[0:1]
    with torch.no_grad():
        for _ in range(10): model_cpu(g0)           # warmup
        t1 = time.perf_counter()
        for _ in range(1000): model_cpu(g0)
        infer_ms = (time.perf_counter() - t1) / 1000 * 1000

    # ── Output ────────────────────────────────────────────────────────────────
    print('\n')
    print(f'{"Metric":<24} {"E1 (IgANets)":<18} {"E4 (FNO ep=1000)"}')
    print('-' * 65)
    print(f'{"Field-space L2":<24} {"3.09%":<18} {mean_l2*100:.2f}%')
    print(f'{"Boundary violation":<24} {"1.51e-4":<18} {max_bc:.2e}')
    print(f'{"Parameter count":<24} {"~580k":<18} {n_params//1000}k  ({n_params:,})')
    print(f'{"Inference time (CPU)":<24} {"0.062 ms":<18} {infer_ms:.3f} ms')
    print(f'{"Trivial baseline":<24} {"11.7%":<18} {trivial*100:.2f}%')
    print(flush=True)


if __name__ == '__main__':
    main()
