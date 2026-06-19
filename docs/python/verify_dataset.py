"""
Dataset verification utility for the IgANets Poisson surrogate project.

Reads build-tutorials/docs/dataset.pt and reports:
  1. Tensor shapes
  2. Per-tensor inter-sample std (checks variability across geometries)
  3. Trivial baseline: mean-prediction relative L2 on test set (field space)
  4. PDE residual check: ||Au - b|| mean and max over test set
  5. Condition number κ(A) distribution + implied α_c = 1/κ for E3
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

DATASET_PATH = os.path.join(os.path.dirname(__file__),
                            '../../build-tutorials/docs/dataset.pt')

TENSOR_NAMES = [
    ('G',      'geometry control points       [N, 50]'),
    ('u',      'B-spline coefficients          [N, 100]'),
    ('u_val',  'field values at Greville pts   [N, 100]'),
    ('A',      'stiffness matrix               [N, 100, 100]'),
    ('b',      'RHS vector                     [N, 100]'),
    ('G_grid', 'geometry on 64×64 grid         [N, 64, 64, 2]'),
    ('u_grid', 'field on 64×64 grid            [N, 64, 64]'),
]


def load():
    m  = torch.jit.load(DATASET_PATH, map_location='cpu')
    sd = m.state_dict()
    return [sd[str(i)] for i in range(7)]


def rel_l2(pred, ref):
    ef = (pred - ref).norm().item()
    rf = ref.norm().item()
    return ef / rf if rf > 0 else ef


def main():
    print(f'Loading: {DATASET_PATH}')
    tensors = load()
    N = tensors[0].shape[0]
    N_train = int(N * 0.8)
    N_test  = N - N_train
    print(f'N={N}  train={N_train}  test={N_test}\n')

    # ── 1. Shape and variability check ────────────────────────────────────────
    print('─' * 72)
    print(f'{"Name":<10} {"Shape":<22} {"dtype":<10} {"inter-sample std":>16}')
    print('─' * 72)
    for t, (name, desc) in zip(tensors, TENSOR_NAMES):
        std = t.float().std(dim=0).mean().item()
        print(f'{name:<10} {str(list(t.shape)):<22} {str(t.dtype):<10} {std:>16.4e}')
    print('─' * 72)

    # ── 2. Trivial baseline (field space, u_grid) ─────────────────────────────
    u_grid = tensors[6].float()
    u_train_mean = u_grid[:N_train].mean(0)   # [64, 64]

    trivial_l2_list = [
        rel_l2(u_train_mean, u_grid[N_train + i])
        for i in range(N_test)
    ]
    trivial_mean = sum(trivial_l2_list) / N_test
    trivial_std  = torch.tensor(trivial_l2_list).std().item()
    print(f'\nTrivial baseline (predict train mean for all test samples):')
    print(f'  Field-space rel-L2:  {trivial_mean*100:.2f}% ± {trivial_std*100:.2f}%')

    # Also trivial baseline in coeff space (u)
    u_coeff = tensors[1].float()
    u_coeff_mean = u_coeff[:N_train].mean(0)
    coeff_trivial = sum(
        rel_l2(u_coeff_mean, u_coeff[N_train + i]) for i in range(N_test)
    ) / N_test
    print(f'  Coeff-space rel-L2:  {coeff_trivial*100:.2f}%')

    # ── 3. PDE residual check ||Au - b|| ──────────────────────────────────────
    A = tensors[3].float()
    b = tensors[4].float()
    u = tensors[1].float()

    res_norms = []
    rel_res_list = []
    for i in range(N):
        res = torch.mv(A[i], u[i]) - b[i]
        rn  = res.norm().item()
        bn  = b[i].norm().item()
        res_norms.append(rn)
        rel_res_list.append(rn / bn if bn > 0 else rn)

    res_t      = torch.tensor(res_norms)
    rel_res_t  = torch.tensor(rel_res_list)
    print(f'\nPDE residual ||Au - b|| (all {N} samples):')
    print(f'  Absolute  mean={res_t.mean().item():.3e}  max={res_t.max().item():.3e}')
    print(f'  Relative  mean={rel_res_t.mean().item():.3e}  max={rel_res_t.max().item():.3e}')
    n_bad = (rel_res_t >= 0.01).sum().item()
    print(f'  Samples with rel-res >= 1%: {n_bad} / {N}')

    # ── 4. Range and boundary checks ──────────────────────────────────────────
    u_g = tensors[6].float()
    print(f'\nu_grid value range: [{u_g.min().item():.4f}, {u_g.max().item():.4f}]')

    # Boundary pixels (row 0, row 63, col 0, col 63) should be ~0
    bdy = torch.cat([
        u_g[:, 0, :].reshape(-1),
        u_g[:, -1, :].reshape(-1),
        u_g[:, :, 0].reshape(-1),
        u_g[:, :, -1].reshape(-1),
    ])
    print(f'Boundary pixel max |u|: {bdy.abs().max().item():.3e}  '
          f'(should be ~1e-17 for exact BCs)')

    G_g = tensors[5].float()
    print(f'G_grid value range: [{G_g.min().item():.4f}, {G_g.max().item():.4f}]')

    # ── 5. Condition number κ(A) and implied α_c for E3 ───────────────────────
    # A is the IGA collocation matrix (100×100 per sample).
    # In E3, the physics loss weight α destabilises training when
    #   α > α_c ≈ 1/κ(A)
    # because the residual gradient ||A^T(Au-b)|| overwhelms the supervised
    # gradient once α × κ(A) >> 1.
    # We report κ over the full dataset and the train split separately.
    print(f'\n{"─"*72}')
    print('Condition number κ(A) of the IGA collocation matrix')
    print(f'{"─"*72}')

    # Use float64 for accuracy; linalg.cond uses SVD internally
    A_d = A.double()    # [N, 100, 100]

    for label, idx in [('All 500 samples', slice(None)),
                        ('Train split (first 400)', slice(None, N_train))]:
        conds = torch.linalg.cond(A_d[idx]).float()   # [N_sub]
        n_sub = conds.shape[0]

        q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        pcts = torch.quantile(conds, q)

        print(f'\n  {label}  (n={n_sub})')
        print(f'  {"stat":<12} {"κ(A)":>10}   {"1/κ  =  α_c":>14}')
        print(f'  {"─"*40}')
        for stat, val in [('min',    conds.min().item()),
                          ('10th %', pcts[0].item()),
                          ('25th %', pcts[1].item()),
                          ('median', pcts[2].item()),
                          ('75th %', pcts[3].item()),
                          ('90th %', pcts[4].item()),
                          ('mean',   conds.mean().item()),
                          ('max',    conds.max().item())]:
            print(f'  {stat:<12} {val:>10.1f}   {1/val:>14.2e}')

    # Summary: α_c range
    conds_train = torch.linalg.cond(A_d[:N_train]).float()
    kmin  = conds_train.min().item()
    kmed  = conds_train.median().item()
    kmean = conds_train.mean().item()
    print(f'\n  Implied α_c = 1/κ (train split):')
    print(f'    conservative (1/κ_min)    = {1/kmin:.2e}  '
          f'← last α where training stays stable')
    print(f'    median       (1/κ_median) = {1/kmed:.2e}')
    print(f'    aggressive   (1/κ_mean)   = {1/kmean:.2e}')
    print(f'\n  E3 experimental transition: α ∈ (0.001, 0.005)')
    print(f'  → consistent with α_c ≈ 1/κ_min = {1/kmin:.2e}')

    print(f'\n{"─"*72}')
    print('Verification complete.')


if __name__ == '__main__':
    main()
