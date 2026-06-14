"""
E4 checkpoint evaluation script.
Trains FNO to 1000 epochs, evaluates at ep=30/100/300/1000.
At each checkpoint computes:
  - train field-L2: actual ||u_pred - u||/||u|| on 400 training samples
  - test  field-L2: actual ||u_pred - u||/||u|| on 100 test samples
Results written to /tmp/e4_table.txt
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

MODES    = 12
WIDTH    = 16
N_LAYERS = 4
BATCH    = 20

CHECKPOINTS = [30, 100, 300, 1000]
RESULT_FILE = '/tmp/e4_table.txt'


def field_l2(model, G, u, device, batch=50):
    model.eval()
    total, n = 0.0, G.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch):
            g = G[i:i+batch].to(device)
            pred = model(g).cpu()
            ref  = u[i:i+batch]
            for j in range(pred.shape[0]):
                ef = (pred[j] - ref[j]).norm().item()
                rf = ref[j].norm().item()
                total += ef / rf if rf > 0 else ef
    model.train()
    return total / n


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    G_all = sd['5'].float()   # [500, 64, 64, 2]
    u_all = sd['6'].float()   # [500, 64, 64]
    N = G_all.shape[0]
    N_tr = int(N * 0.8)

    G_tr, u_tr = G_all[:N_tr], u_all[:N_tr]
    G_te, u_te = G_all[N_tr:], u_all[N_tr:]

    loader = DataLoader(TensorDataset(G_tr, u_tr),
                        batch_size=BATCH, shuffle=True)

    model = FNO2d(MODES, MODES, WIDTH, N_LAYERS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    rows   = []
    ep     = 0
    ck_idx = 0
    t_start = time.time()

    # Open result file and write header
    with open(RESULT_FILE, 'w') as f:
        f.write('checkpoint,train_l2,test_l2,elapsed_s\n')

    while ck_idx < len(CHECKPOINTS):
        target = CHECKPOINTS[ck_idx]

        # Train up to target
        while ep < target:
            ep_loss = 0.0
            for G_b, u_b in loader:
                G_b, u_b = G_b.to(device), u_b.to(device)
                opt.zero_grad()
                loss = F.mse_loss(model(G_b), u_b)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            ep += 1

        # Evaluate
        tr_l2 = field_l2(model, G_tr, u_tr, device)
        te_l2 = field_l2(model, G_te, u_te, device)
        elapsed = time.time() - t_start

        row = f'{target},{tr_l2*100:.3f},{te_l2*100:.3f},{elapsed:.1f}'
        rows.append(row)
        with open(RESULT_FILE, 'a') as f:
            f.write(row + '\n')

        print(f'ep={target:4d}  train={tr_l2*100:.3f}%  test={te_l2*100:.3f}%  '
              f't={elapsed:.0f}s  mse={ep_loss/len(loader):.2e}', flush=True)

        ck_idx += 1

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
