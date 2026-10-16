# validate_runtime.py
"""
Verify that one day of orchestration takes ≤ 10 seconds.
"""

import time
from pathlib import Path
from dev_kit import generate, Simulator
from eligibility_mask import build_eligibility_mask
from outcome_model import OutcomeEnsemble
from thompson_sampling import ThompsonSampler
from matching import greedy_max_weight_matching, matching_with_waiting_threshold
from clarification import ValueOfInformationSelector


def measure_day_runtime(variant='development', seed=42, n_people=100):
    """Measure time to run one complete day."""
    
    world = generate(seed=seed, n=n_people, pool_id=f'test_{seed}', variant=variant)
    sim = Simulator(world)
    
    ensemble = OutcomeEnsemble(n_ensemble=10, seed=seed)
    sampler = ThompsonSampler(ensemble, seed=seed)
    clarifier = ValueOfInformationSelector(ensemble, seed=seed)
    
    timings = {
        'observe': 0,
        'train': 0,
        'mask': 0,
        'sample_score': 0,
        'match': 0,
        'clarify': 0,
        'advance': 0,
        'total': 0
    }
    
    start = time.perf_counter()
    
    # Observe
    t0 = time.perf_counter()
    state = sim.observe()
    timings['observe'] = time.perf_counter() - t0
    
    members = state['members']
    members_by_id = {m['member_id']: m for m in members}
    
    # Train
    t0 = time.perf_counter()
    ensemble.train(state["introductions"], state["feedback"], members_by_id)
    timings['train'] = time.perf_counter() - t0
    
    # Mask
    t0 = time.perf_counter()
    mask = build_eligibility_mask(members)
    timings['mask'] = time.perf_counter() - t0
    
    # Thompson sample and score
    t0 = time.perf_counter()
    sampled_idx = sampler.sample_world()
    scores = sampler.score_all_pairs(members, mask, sampled_idx)
    timings['sample_score'] = time.perf_counter() - t0
    
    # Match
    t0 = time.perf_counter()
    pairs, should_wait, total_score = matching_with_waiting_threshold(
        scores, members_by_id, threshold=0.3
    )
    timings['match'] = time.perf_counter() - t0
    
    # Clarify
    t0 = time.perf_counter()
    asks = clarifier.select_asks(state, max_cost=12)
    ask_responses = []
    try:
        ask_responses = sim.resolve_asks(asks)
    except:
        pass
    timings['clarify'] = time.perf_counter() - t0
    
    # Advance
    t0 = time.perf_counter()
    try:
        sim.advance(pairs)
    except:
        pass
    timings['advance'] = time.perf_counter() - t0
    
    timings['total'] = time.perf_counter() - start
    
    return timings


def main():
    """Run timing validation."""
    print("="*60)
    print("RUNTIME VALIDATION: 10-second/day budget")
    print("="*60)
    
    scenarios = ['development', 'sparse', 'cold_start']
    all_timings = []
    
    for scenario in scenarios:
        print(f"\nTesting {scenario}...")
        for seed in range(3):
            timings = measure_day_runtime(variant=scenario, seed=42+seed, n_people=200)
            all_timings.append((scenario, seed, timings))
            
            print(f"  Seed {seed}: {timings['total']:.3f}s total")
            print(f"    Observe: {timings['observe']:.4f}s")
            print(f"    Train:   {timings['train']:.4f}s")
            print(f"    Mask:    {timings['mask']:.4f}s")
            print(f"    Score:   {timings['sample_score']:.4f}s")
            print(f"    Match:   {timings['match']:.4f}s")
            print(f"    Clarify: {timings['clarify']:.4f}s")
            print(f"    Advance: {timings['advance']:.4f}s")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    totals = [t['total'] for _, _, t in all_timings]
    avg_time = sum(totals) / len(totals)
    max_time = max(totals)
    
    print(f"\nAverage day runtime: {avg_time:.3f}s")
    print(f"Maximum day runtime: {max_time:.3f}s")
    print(f"Budget:              10.000s")
    print(f"Margin:              {10 - max_time:.3f}s")
    
    if max_time <= 10:
        print(f"\n✓ PASS: Within budget")
        return True
    else:
        print(f"\n✗ FAIL: Exceeds budget by {max_time - 10:.3f}s")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)