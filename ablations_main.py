# ablations_main.py
"""
Run one complete episode (60 days + 40 days of feedback settling) with an
ablation flag. Results are written to:

    <output_root>/<ablation>/<variant>/<variant>_<seed>.json

Ablations (each removes one component):
    full                  all components
    no_clarification      never ask questions
    random_clarification  ask 4 random members about constraints (not VOI)
    greedy_ranking        greedy matching without the waiting threshold
    epsilon_greedy        always score with ensemble member 0 (no Thompson)
    no_discounting        train on ALL history, not just recent outcomes
    no_waiting_threshold  always match, whatever the scores
    no_random_slice       placeholder (random slice not implemented yet)

All ablations obey the slot rule: one active introduction per person (7 days).
    soft_penalty_mask     no hard mask: every pair treated as feasible
"""

import json
import random
from pathlib import Path

from dev_kit import generate, Simulator
from eligibility_mask import build_eligibility_mask, check_hard_constraints
from outcome_model import OutcomeEnsemble
from thompson_sampling import ThompsonSampler
from matching import greedy_max_weight_matching, matching_with_waiting_threshold
from clarification import ValueOfInformationSelector

ABLATIONS = [
    'full', 'no_clarification', 'random_clarification', 'greedy_ranking',
    'epsilon_greedy', 'no_discounting', 'no_waiting_threshold',
    'no_random_slice', 'soft_penalty_mask',
]
ALIASES = {'no_thompson': 'epsilon_greedy'}   # old name used by earlier scripts

WAIT_THRESHOLD = 0.3
SLOT_DAYS = 7              # an introduction occupies both people's single slot for 7 days (proposal, Sec. 4.1)
RECENT_WINDOW = 400        # outcomes used for training (discounting)
ALL_HISTORY = 100000       # "no discounting": effectively unlimited


def run_episode_with_ablations(variant='development', seed=42, n_people=200,
                               days=60, ablation='full', output_dir=None,
                               verbose=False, n_ensemble=10):
    ablation = ALIASES.get(ablation, ablation)
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation '{ablation}'. Choose from {ABLATIONS}")

    output_dir = Path(output_dir) if output_dir else Path('results') / ablation / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- world ----------
    world = generate(seed=seed, n=n_people, pool_id=f'{variant}_{seed}', variant=variant)
    sim = Simulator(world)

    # ---------- flags ----------
    use_clarification = ablation != 'no_clarification'
    use_voi_clarification = ablation not in ('no_clarification', 'random_clarification')
    use_waiting_threshold = ablation not in ('no_waiting_threshold', 'greedy_ranking')
    use_thompson_sampling = ablation != 'epsilon_greedy'
    use_hard_mask = ablation != 'soft_penalty_mask'
    use_discounting = ablation != 'no_discounting'

    # ---------- models ----------
    ensemble = OutcomeEnsemble(
        n_ensemble=n_ensemble, seed=seed,
        max_obs=RECENT_WINDOW if use_discounting else ALL_HISTORY)
    sampler = ThompsonSampler(ensemble, seed=seed)
    clarifier = ValueOfInformationSelector(ensemble, seed=seed)
    clar_rng = random.Random(seed + 12345)     # local RNG: never touch global state
    busy_until = {}                            # member_id -> first day their slot is free again

    logs = {
        'variant': variant, 'seed': seed, 'ablation': ablation,
        'n_people': n_people, 'days': days,
        'ablation_flags': {
            'use_clarification': use_clarification,
            'use_voi_clarification': use_voi_clarification,
            'use_waiting_threshold': use_waiting_threshold,
            'use_thompson_sampling': use_thompson_sampling,
            'use_hard_mask': use_hard_mask,
            'use_discounting': use_discounting,
        },
        'daily_logs': [],
    }

    for day_num in range(days):
        state = sim.observe()
        members = state['members']
        members_by_id = {m['member_id']: m for m in members}

        # 1. learn from everything observed so far
        ensemble.train(state['introductions'], state['feedback'], members_by_id)

        # 2. hard-constraint mask (or "everything feasible" for the soft ablation)
        if use_hard_mask:
            mask = build_eligibility_mask(members)
        else:
            mask = {(a['member_id'], b['member_id']): 1
                    for i, a in enumerate(members) for b in members[i + 1:]}

        # 3. score pairs (Thompson sample, or fixed member 0)
        idx = sampler.sample_world() if use_thompson_sampling else 0
        scores = sampler.score_all_pairs(members, mask, idx)

        # 4. hard rules the simulator does not enforce:
        #    - a pair is only ever introduced once
        #    - each person holds at most one active introduction (7-day slot)
        introduced = {tuple(sorted((a, b))) for a, b, _ in state['introductions']}
        scores = {p: s for p, s in scores.items()
                  if p not in introduced
                  and busy_until.get(p[0], 0) <= day_num
                  and busy_until.get(p[1], 0) <= day_num}

        # 5. match
        if use_waiting_threshold:
            pairs, should_wait, match_score = matching_with_waiting_threshold(
                scores, members_by_id, threshold=WAIT_THRESHOLD)
        else:
            pairs = greedy_max_weight_matching(scores, members_by_id)
            should_wait = len(pairs) == 0
            match_score = sum(scores.get(p, 0.0) for p in pairs)

        # 6. clarification questions
        ask_responses = []
        if use_clarification:
            if use_voi_clarification:
                asks = clarifier.select_asks(state, max_cost=12)
            else:
                eligible = [m for m in members if m.get('available', True)]
                chosen = clar_rng.sample(eligible, min(4, len(eligible)))
                asks = [{'member_id': m['member_id'], 'field': 'constraints'} for m in chosen]
            ask_responses = sim.resolve_asks(asks)

        # 7. audit BEFORE advancing, then advance
        _valid, violations = check_hard_constraints(state, pairs)
        sim.advance(pairs)
        for a_id, b_id in pairs:
            busy_until[a_id] = busy_until[b_id] = day_num + SLOT_DAYS

        logs['daily_logs'].append({
            'day': day_num,
            'pairs_introduced': len(pairs),
            'total_score': match_score,
            'asks_made': len(ask_responses),
            'rule_violations': len(violations),
            'should_wait': should_wait,
        })
        sampler.advance_day()

        if verbose:
            print(f"  day {day_num:2d}: pairs={len(pairs):3d} "
                  f"avg_score={match_score / max(len(pairs), 1):.3f} "
                  f"violations={len(violations)} wait={should_wait}")

    # let late feedback arrive
    for _ in range(40):
        sim.advance([])

    logs['final_metrics'] = sim.metrics()
    log_path = output_dir / f"{variant}_{seed}.json"
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    m = logs['final_metrics']
    print(f"[{ablation:22s}] {variant:12s} seed={seed:3d}  "
          f"MSMI/100={m['msmi_per_100_arrived_members']:5.1f}  "
          f"intros={m['total_introductions']:5d}  "
          f"violations={sum(d['rule_violations'] for d in logs['daily_logs'])}")
    return logs


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Run one episode with an ablation flag')
    p.add_argument('--variant', default='development')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n', type=int, default=200, help='pool size')
    p.add_argument('--days', type=int, default=60)
    p.add_argument('--ablation', default='full', choices=ABLATIONS + list(ALIASES))
    p.add_argument('--output_root', default='results',
                   help='root folder; file goes to <root>/<ablation>/<variant>/')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()

    ab = ALIASES.get(a.ablation, a.ablation)
    run_episode_with_ablations(
        variant=a.variant, seed=a.seed, n_people=a.n, days=a.days, ablation=ab,
        output_dir=Path(a.output_root) / ab / a.variant, verbose=a.verbose)
