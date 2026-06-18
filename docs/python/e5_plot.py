"""
E5 — Generalization study: plots from results.csv.

Plot A: Field-L2 vs σ_test  (N_train=400, E1 vs E2)
         Shows extrapolation behaviour as geometry complexity grows.

Plot B: Field-L2 vs N_train  (σ_test=0.10, E1 vs E2)
         Shows data-efficiency comparison.

Output: docs/python/e5_plot_A.pdf  and  docs/python/e5_plot_B.pdf
"""

import sys, os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'results.csv')
OUT_A       = os.path.join(os.path.dirname(__file__), 'e5_plot_A.pdf')
OUT_B       = os.path.join(os.path.dirname(__file__), 'e5_plot_B.pdf')

COLORS = {'E1': '#2166ac', 'E2': '#d6604d'}   # blue / red


def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append({
                'model':      r['model'],
                'N_train':    int(r['N_train']),
                'sigma_test': float(r['sigma_test']),
                'field_l2':   float(r['field_l2']) * 100,   # percent
                'trivial_l2': float(r['trivial_l2']) * 100,
            })
    return rows


def get(rows, model, key_name, key_val, metric):
    return [r[metric] for r in rows
            if r['model'] == model and r[key_name] == key_val]


def get_x(rows, model, key_name, key_val, x_name):
    return [r[x_name] for r in rows
            if r['model'] == model and r[key_name] == key_val]


# ── Plot A: Field-L2 vs σ_test, N_train=400 ──────────────────────────────────

def plot_A(rows):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for model in ('E1', 'E2'):
        xs = sorted(set(get_x(rows, model, 'N_train', 400, 'sigma_test')))
        ys = [next(r['field_l2'] for r in rows
                   if r['model']==model and r['N_train']==400
                   and r['sigma_test']==x) for x in xs]
        ax.plot(xs, ys, 'o-', color=COLORS[model], label=model, linewidth=1.8,
                markersize=6)

    # Trivial baseline at N=400 (take E1's values, identical for both)
    xs = sorted(set(get_x(rows, 'E1', 'N_train', 400, 'sigma_test')))
    ys_triv = [next(r['trivial_l2'] for r in rows
                    if r['model']=='E1' and r['N_train']==400
                    and r['sigma_test']==x) for x in xs]
    ax.plot(xs, ys_triv, '--', color='gray', label='Trivial baseline',
            linewidth=1.2)

    ax.set_xlabel('Geometry perturbation σ (test set)', fontsize=11)
    ax.set_ylabel('Field-space relative L2 (%)', fontsize=11)
    ax.set_title('E5 — Generalisation: field error vs geometry complexity\n'
                 '(N_train = 400)', fontsize=10)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_A, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT_A}')


# ── Plot B: Field-L2 vs N_train, σ_test=0.10 ─────────────────────────────────

def plot_B(rows):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for model in ('E1', 'E2'):
        xs = sorted(set(get_x(rows, model, 'sigma_test', 0.10, 'N_train')))
        ys = [next(r['field_l2'] for r in rows
                   if r['model']==model and r['sigma_test']==0.10
                   and r['N_train']==x) for x in xs]
        ax.plot(xs, ys, 'o-', color=COLORS[model], label=model, linewidth=1.8,
                markersize=6)

    # Trivial baseline at σ=0.10 (changes with N_train because train mean shifts)
    xs = sorted(set(get_x(rows, 'E1', 'sigma_test', 0.10, 'N_train')))
    ys_triv = [next(r['trivial_l2'] for r in rows
                    if r['model']=='E1' and r['sigma_test']==0.10
                    and r['N_train']==x) for x in xs]
    ax.plot(xs, ys_triv, '--', color='gray', label='Trivial baseline',
            linewidth=1.2)

    ax.set_xlabel('Training set size N_train', fontsize=11)
    ax.set_ylabel('Field-space relative L2 (%)', fontsize=11)
    ax.set_title('E5 — Data efficiency: field error vs training size\n'
                 '(σ_test = 0.10)', fontsize=10)
    ax.set_xscale('log')
    ax.set_xticks([50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_B, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT_B}')


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f'ERROR: {RESULTS_CSV} not found. Run e5_generalization.py first.')
        sys.exit(1)

    rows = load_csv(RESULTS_CSV)
    print(f'Loaded {len(rows)} rows from {RESULTS_CSV}')

    plot_A(rows)
    plot_B(rows)
    print('Done.')


if __name__ == '__main__':
    main()
