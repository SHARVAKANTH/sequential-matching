# plot_results.py
"""
Generates three charts from results/<ablation>/<variant>/*.json:
  results/comparison.png  - MSMI, introductions and rule violations per ablation
  results/ablations.png   - % MSMI change when each component is removed
  results/hypotheses.png  - MSMI difference vs full, with standard error

Averages over every seed found for the chosen scenario (default: development).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ablations_main import ABLATIONS

RESULTS = Path('results')
COMPONENT = {
    'no_clarification': 'VOI clarification (none)',
    'random_clarification': 'VOI clarification (random)',
    'greedy_ranking': 'Waiting threshold',
    'epsilon_greedy': 'Thompson sampling',
    'no_discounting': 'Discounting',
    'no_waiting_threshold': 'Waiting threshold',
    'no_random_slice': 'Random slice (placeholder)',
    'soft_penalty_mask': 'Hard constraint mask',
}


def load_results(variant):
    out = {}
    for ab in ABLATIONS:
        eps = []
        for f in sorted((RESULTS / ab / variant).glob('*.json')):
            with open(f) as fh:
                eps.append(json.load(fh))
        if eps:
            out[ab] = eps
    return out


def msmi(e): return e['final_metrics']['msmi_per_100_arrived_members']
def intros(e): return e['final_metrics']['total_introductions']
def viol(e): return sum(d.get('rule_violations', 0) for d in e['daily_logs'])


def mean_se(eps, fn):
    v = np.array([fn(e) for e in eps], dtype=float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), se


def plot_comparison(res, variant):
    names = list(res)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, fn, title, ylabel, color in [
        (axes[0], msmi, 'MSMI performance', 'MSMI per 100 members', '#2ecc71'),
        (axes[1], intros, 'Throughput', 'Total introductions', '#e74c3c'),
        (axes[2], viol, 'Constraint compliance', 'Rule violations (per episode)', '#f39c12'),
    ]:
        stats = [mean_se(res[a], fn) for a in names]
        ax.bar(range(len(names)), [s[0] for s in stats], yerr=[s[1] for s in stats],
               color=color, edgecolor='black', capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([a.replace('_', '\n') for a in names], fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        top = max(s[0] + s[1] for s in stats) or 1
        for i, (m, _) in enumerate(stats):
            ax.text(i, m + top * 0.02, f'{m:.0f}' if m >= 100 else f'{m:.1f}', ha='center', fontsize=8)
    n = min(len(v) for v in res.values())
    plt.suptitle(f'Ablation study: direct comparison ({variant}, {n}+ seeds, error bars = s.e.)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("saved results/comparison.png")


def plot_ablation_impact(res):
    if 'full' not in res:
        print("skip ablations.png (no 'full' baseline)")
        return
    base = np.mean([msmi(e) for e in res['full']])
    names = [a for a in res if a != 'full']
    impact = [(base - np.mean([msmi(e) for e in res[a]])) / base * 100 for a in names]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if x > 5 else '#f39c12' if x > 0 else '#3498db' for x in impact]
    bars = ax.barh([f"{a}\n({COMPONENT.get(a, a)})" for a in names], impact,
                   color=colors, edgecolor='black')
    for bar, v in zip(bars, impact):
        ax.text(v + (0.2 if v >= 0 else -0.2), bar.get_y() + bar.get_height() / 2, f'{v:.1f}%',
                va='center', ha='left' if v >= 0 else 'right', fontweight='bold')
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Full system')
    ax.set_xlabel('MSMI lost when component is removed (%); negative = ablation scored HIGHER',
                  fontweight='bold')
    ax.set_title('Component contribution to MSMI', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / 'ablations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("saved results/ablations.png")


def plot_hypotheses(res):
    if 'full' not in res:
        print("skip hypotheses.png (no 'full' baseline)")
        return
    tests = [('H1\nThompson vs\nepsilon-greedy', 'epsilon_greedy'),
             ('H2\nThreshold vs\ngreedy ranking', 'greedy_ranking'),
             ('H3\nVOI vs none', 'no_clarification'),
             ('H3\nVOI vs random', 'random_clarification'),
             ('H6\nHard mask vs\nsoft penalty', 'soft_penalty_mask')]
    tests = [(l, a) for l, a in tests if a in res]
    full = np.array([msmi(e) for e in res['full']])
    labels, diffs, ses = [], [], []
    for label, a in tests:
        other = np.array([msmi(e) for e in res[a]])
        k = min(len(full), len(other))
        d = full[:k] - other[:k]                     # paired by seed
        labels.append(label)
        diffs.append(d.mean())
        ses.append(d.std(ddof=1) / np.sqrt(k) if k > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(labels)), diffs, yerr=ses, capsize=4, edgecolor='black',
           color=['#2ecc71' if d > 0 else '#e74c3c' for d in diffs])
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MSMI per 100: full minus ablation (paired by seed)', fontweight='bold')
    ax.set_title('Hypothesis view: positive = full system better', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(RESULTS / 'hypotheses.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("saved results/hypotheses.png")


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'development'
    res = load_results(variant)
    if not res:
        print(f"No results in results/<ablation>/{variant}/. Run: python experiment_runner.py --small")
        return
    print(f"Loaded {len(res)} ablations for '{variant}': " +
          ', '.join(f"{a}({len(v)})" for a, v in res.items()))
    RESULTS.mkdir(exist_ok=True)
    plot_comparison(res, variant)
    plot_ablation_impact(res)
    plot_hypotheses(res)


if __name__ == '__main__':
    main()
