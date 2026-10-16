# experiment_runner.py
"""
Batch executor: scenarios x seeds x ablations, in parallel.
Full sweep = 6 scenarios x 20 seeds x 9 ablations = 1,080 episodes.

Output: results/<ablation>/<variant>/<variant>_<seed>.json
Existing files are skipped, so an interrupted sweep can be resumed.
"""

import argparse
from pathlib import Path
from multiprocessing import Pool

from ablations_main import run_episode_with_ablations, ABLATIONS

SCENARIOS = ['development', 'sparse', 'cold_start', 'delayed', 'shift', 'drift']


def run_single_experiment(job):
    variant, seed, ablation, root = job
    out_dir = Path(root) / ablation / variant
    if (out_dir / f'{variant}_{seed}.json').exists():
        return f"= {variant:12} seed={seed:3} {ablation:22} (exists, skipped)"
    try:
        run_episode_with_ablations(variant=variant, seed=seed, n_people=200, days=60,
                                   ablation=ablation, output_dir=out_dir)
        return f"v {variant:12} seed={seed:3} {ablation:22}"
    except Exception as e:
        return f"X {variant:12} seed={seed:3} {ablation:22} ERROR: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--small', action='store_true', help='2 scenarios x 2 seeds')
    p.add_argument('--seeds', type=int, default=None, help='number of seeds (default 20, or 2 with --small)')
    p.add_argument('--output_root', default='results')
    args = p.parse_args()

    scenarios = SCENARIOS[:2] if args.small else SCENARIOS
    n_seeds = args.seeds or (2 if args.small else 20)
    seeds = range(42, 42 + n_seeds)

    jobs = [(v, s, a, args.output_root) for v in scenarios for s in seeds for a in ABLATIONS]
    print(f"Running {len(jobs)} episodes with {args.workers} workers...")

    failures = 0
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_single_experiment, jobs), 1):
            failures += res.startswith('X')
            if res.startswith('X') or i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {res}")
    print(f"\nCompleted {len(jobs)} episodes, {failures} failed")


if __name__ == '__main__':
    main()
