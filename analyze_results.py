# analyze_results.py
"""
Load episode results, compute bootstrap CIs, test hypotheses H1, H2, H3, H6.

Missing data is reported as "n/a" and the hypothesis is SKIPPED. (Earlier
versions turned empty data into 0.0, which looked like a real 0.0 result.)
"""

import json
import random
import statistics
from pathlib import Path

from ablations_main import ABLATIONS

RESULTS = Path('results')
SCENARIOS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


# ------------------------------------------------------------------ loading
def load_episodes(ablation, variant, root=RESULTS):
    d = Path(root) / ablation / variant
    eps = []
    if d.exists():
        for f in sorted(d.glob('*.json')):
            try:
                with open(f) as fh:
                    eps.append(json.load(fh))
            except (OSError, json.JSONDecodeError):
                print(f"  warning: could not read {f}")
    return eps


def msmi_values(episodes):
    return [e['final_metrics']['msmi_per_100_arrived_members']
            for e in episodes if 'final_metrics' in e]


def violation_totals(episodes):
    """Total rule violations per episode (daily value is an int; lists also accepted)."""
    out = []
    for e in episodes:
        total = 0
        for day in e.get('daily_logs', []):
            v = day.get('rule_violations', 0)
            total += len(v) if isinstance(v, (list, tuple)) else int(v)
        out.append(total)
    return out


# ------------------------------------------------------------------ stats
def bootstrap_ci(values, confidence=0.95, n_resamples=1000, seed=0):
    """Bootstrap CI for the mean. Returns None when there is no data."""
    if not values:
        return None
    mean = statistics.mean(values)
    if len(values) < 2:
        return {'mean': mean, 'ci_lower': mean, 'ci_upper': mean, 'std': 0.0, 'n': 1}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                   for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return {'mean': mean,
            'ci_lower': means[int(alpha * n_resamples)],
            'ci_upper': means[min(int((1 - alpha) * n_resamples), n_resamples - 1)],
            'std': statistics.stdev(values), 'n': len(values)}


def fmt(ci):
    if ci is None:
        return 'n/a (no data)'
    return f"{ci['mean']:.2f} [{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}] n={ci['n']}"


def pick_scenario(data, ablations, preferred):
    """Use the preferred scenario if every ablation has data there, else 'development'."""
    for sc in (preferred, 'development'):
        if all(data[a][sc] for a in ablations):
            return sc
    return None


def skipped(name, reason):
    return {'hypothesis': name, 'skipped': True, 'reason': reason}


# ------------------------------------------------------------------ hypotheses
def test_h1(data):
    name = 'H1: Thompson > epsilon-greedy, biggest gap in cold_start'
    sc = pick_scenario(data, ['full', 'epsilon_greedy'], 'cold_start')
    if sc is None:
        return skipped(name, 'no episodes for full and epsilon_greedy')
    t, g = bootstrap_ci(msmi_values(data['full'][sc])), bootstrap_ci(msmi_values(data['epsilon_greedy'][sc]))
    diff = t['mean'] - g['mean']
    return {'hypothesis': name, 'scenario_used': sc, 'thompson': t, 'epsilon_greedy': g,
            'difference': diff, 'pct_gain': diff / g['mean'] * 100 if g['mean'] else None,
            'passes': diff > 0 and t['ci_lower'] > g['ci_upper']}


def test_h2(data):
    name = 'H2: global matching > greedy ranking, smallest gap in sparse'
    sc = pick_scenario(data, ['full', 'greedy_ranking'], 'sparse')
    if sc is None:
        return skipped(name, 'no episodes for full and greedy_ranking')
    m, g = bootstrap_ci(msmi_values(data['full'][sc])), bootstrap_ci(msmi_values(data['greedy_ranking'][sc]))
    diff = m['mean'] - g['mean']
    return {'hypothesis': name, 'scenario_used': sc, 'matching': m, 'greedy_ranking': g,
            'difference': diff, 'passes': diff > 0 and m['ci_lower'] > g['ci_upper']}


def test_h3(data):
    name = 'H3: VoI clarification > no clarification and > random clarification'
    abl = ['full', 'no_clarification', 'random_clarification']
    sc = pick_scenario(data, abl, 'development')
    if sc is None:
        return skipped(name, 'missing episodes for ' + ', '.join(abl))
    f, n, r = (bootstrap_ci(msmi_values(data[a][sc])) for a in abl)
    return {'hypothesis': name, 'scenario_used': sc, 'full': f, 'no_clarification': n,
            'random_clarification': r,
            'gain_vs_none': f['mean'] - n['mean'], 'gain_vs_random': f['mean'] - r['mean'],
            'passes': f['mean'] > n['mean'] and f['mean'] > r['mean']}


def test_h6(data):
    name = 'H6: hard mask -> zero violations and MSMI >= soft penalty'
    sc = pick_scenario(data, ['full', 'soft_penalty_mask'], 'development')
    if sc is None:
        return skipped(name, 'no episodes for full and soft_penalty_mask')
    hard, soft = data['full'][sc], data['soft_penalty_mask'][sc]
    h_ci, s_ci = bootstrap_ci(msmi_values(hard)), bootstrap_ci(msmi_values(soft))
    hv, sv = sum(violation_totals(hard)), sum(violation_totals(soft))
    return {'hypothesis': name, 'scenario_used': sc, 'hard_violations': hv, 'soft_violations': sv,
            'hard_msmi': h_ci, 'soft_msmi': s_ci,
            'zero_violations': hv == 0, 'msmi_at_least_soft': h_ci['mean'] >= s_ci['ci_lower'],
            'passes': hv == 0 and h_ci['mean'] >= s_ci['ci_lower']}


# ------------------------------------------------------------------ main
def print_result(r, lines):
    print(f"\n{r['hypothesis']}")
    if r.get('skipped'):
        print(f"  SKIPPED: {r['reason']}")
        return
    print(f"  scenario: {r['scenario_used']}")
    for label, key in lines:
        v = r[key]
        print(f"  {label:<24}{fmt(v) if isinstance(v, dict) else v}")
    print(f"  PASS: {r['passes']}")


def main():
    print("Loading results...")
    data = {a: {sc: load_episodes(a, sc) for sc in SCENARIOS} for a in ABLATIONS}
    scenarios = [sc for sc in SCENARIOS if any(data[a][sc] for a in ABLATIONS)]
    if not scenarios:
        print("No results found. Run: python experiment_runner.py --small")
        return
    for sc in scenarios:
        print(f"  {sc:12} " + ' '.join(f"{len(data[a][sc]):2d}" for a in ABLATIONS))
    print("  (columns: " + ', '.join(ABLATIONS) + ")")

    print("\n" + "=" * 60 + "\nHYPOTHESIS TESTS\n" + "=" * 60)
    h = {'H1': test_h1(data), 'H2': test_h2(data), 'H3': test_h3(data), 'H6': test_h6(data)}
    print_result(h['H1'], [('Thompson', 'thompson'), ('Epsilon-greedy', 'epsilon_greedy'),
                           ('Difference', 'difference')])
    print_result(h['H2'], [('Global matching', 'matching'), ('Greedy ranking', 'greedy_ranking'),
                           ('Difference', 'difference')])
    print_result(h['H3'], [('Full (VoI)', 'full'), ('No clarification', 'no_clarification'),
                           ('Random clarification', 'random_clarification'),
                           ('Gain vs none', 'gain_vs_none'), ('Gain vs random', 'gain_vs_random')])
    print_result(h['H6'], [('Hard violations', 'hard_violations'), ('Soft violations', 'soft_violations'),
                           ('Hard MSMI', 'hard_msmi'), ('Soft MSMI', 'soft_msmi'),
                           ('Zero violations', 'zero_violations'), ('MSMI >= soft', 'msmi_at_least_soft')])

    print("\n" + "=" * 60 + "\nMSMI per 100 members: mean [95% CI]\n" + "=" * 60)
    summary = {}
    print(f"{'Method':<24}" + ''.join(f"{sc:<26}" for sc in scenarios))
    for a in ABLATIONS:
        summary[a] = {sc: bootstrap_ci(msmi_values(data[a][sc])) for sc in scenarios}
        row = f"{a:<24}"
        for sc in scenarios:
            ci = summary[a][sc]
            row += f"{'n/a':<26}" if ci is None else f"{ci['mean']:>6.1f} [{ci['ci_lower']:>6.1f},{ci['ci_upper']:>6.1f}]  "
        print(row)

    print("\n" + "=" * 60 + "\nTotal rule violations (all episodes)\n" + "=" * 60)
    for a in ABLATIONS:
        tot = {sc: sum(violation_totals(data[a][sc])) for sc in scenarios if data[a][sc]}
        print(f"{a:<24}{tot}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'analysis.json', 'w') as f:
        json.dump({'hypotheses': h, 'summary_table': summary}, f, indent=2, default=str)
    print(f"\nSaved {RESULTS / 'analysis.json'}")


if __name__ == '__main__':
    main()
