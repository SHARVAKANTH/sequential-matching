# test_quick.py
"""
Quick sanity check with detailed diagnostics.
"""

import sys
from dev_kit import generate, Simulator
from eligibility_mask import build_eligibility_mask
from outcome_model import OutcomeEnsemble
from thompson_sampling import ThompsonSampler
from matching import greedy_max_weight_matching
from clarification import ValueOfInformationSelector


def test_basic_flow():
    """Test basic orchestration with diagnostics."""
    print("Testing with diagnostics...\n")
    
    try:
        # Generate
        print("✓ Generating pool...")
        world = generate(seed=42, n=50, pool_id='test_01', variant='development')
        sim = Simulator(world)
        
        # Initialize
        print("✓ Initializing models...")
        ensemble = OutcomeEnsemble(n_ensemble=5, seed=42)
        sampler = ThompsonSampler(ensemble, seed=42)
        clarifier = ValueOfInformationSelector(ensemble, seed=42)
        
        # Just ONE day for diagnostics
        print("\n✓ Day 1/1 (diagnostic run)...\n")
        
        # Observe
        state = sim.observe()
        members = state['members']
        members_by_id = {m['member_id']: m for m in members}
        
        print(f"  Members loaded: {len(members)}")
        
        # Show sample member
        if members:
            sample = members[0]
            print(f"  Sample member: {sample}")
        
        # Train
        print("\n  Training ensemble...")
        ensemble.train(state["introductions"], state["feedback"], members_by_id)
        
        # Soft mask
        print("  Building soft mask...")
        mask = {
            (a['member_id'], b['member_id']): 1
            for a in members
            for b in members
            if a['member_id'] < b['member_id']
        }
        print(f"  Mask size: {len(mask)}")
        
        # Sample
        print("\n  Sampling ensemble member...")
        sampled_idx = sampler.sample_world()
        print(f"  Sampled index: {sampled_idx}")
        
        # TEST: Get one prediction manually
        print("\n  ===== DIAGNOSTIC: Manual Prediction =====")
        if len(members) >= 2:
            person_a = members[0]
            person_b = members[1]
            print(f"  Person A: {person_a['member_id']} age={person_a.get('age')} gender={person_a.get('gender')}")
            print(f"  Person B: {person_b['member_id']} age={person_b.get('age')} gender={person_b.get('gender')}")
            
            # Get ensemble prediction
            pred, std = ensemble.predict_msmi_ensemble(person_a, person_b)
            print(f"  Ensemble prediction: {pred:.4f} ± {std:.4f}")
            
            # Get single model prediction (like Thompson would)
            pred_single = ensemble.predict_msmi(person_a, person_b, sampled_idx)
            print(f"  Sampled model prediction: {pred_single:.4f}")
        
        # Score all pairs
        print("\n  ===== SCORING =====")
        print("  Calling score_all_pairs()...")
        scores = sampler.score_all_pairs(members, mask, sampled_idx)
        
        print(f"  Total scores returned: {len(scores)}")
        
        if scores:
            score_list = list(scores.values())
            min_s = min(score_list)
            max_s = max(score_list)
            avg_s = sum(score_list) / len(score_list)
            pos_s = sum(1 for s in score_list if s > 0)
            zero_s = sum(1 for s in score_list if s == 0)
            neg_s = sum(1 for s in score_list if s < 0)
            
            print(f"  Score distribution:")
            print(f"    Min: {min_s:.4f}")
            print(f"    Max: {max_s:.4f}")
            print(f"    Avg: {avg_s:.4f}")
            print(f"    Positive: {pos_s}")
            print(f"    Zero: {zero_s}")
            print(f"    Negative: {neg_s}")
            
            # Show top 5 scores
            top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  Top 5 scores:")
            for (a, b), score in top_5:
                print(f"    ({a}, {b}): {score:.4f}")
        else:
            print(f"  ✗ NO SCORES RETURNED!")
        
        # Match
        print("\n  ===== MATCHING =====")
        pairs = greedy_max_weight_matching(scores, members_by_id)
        print(f"  Pairs returned: {len(pairs)}")
        
        if pairs:
            print(f"  Sample pairs: {pairs[:5]}")
        
        # Advance
        if pairs:
            sim.advance(pairs)
            print(f"\n  Advanced simulation with {len(pairs)} pairs")
        
        sampler.advance_day()
        
        # Quick finalize
        print("\n  Finalizing (20 days)...")
        for _ in range(20):
            sim.advance([])
        
        metrics = sim.metrics()
        
        print(f"\n  ====== RESULTS ======")
        print(f"  Total Introductions: {metrics['total_introductions']}")
        print(f"  MSMI per 100: {metrics['msmi_per_100_arrived_members']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_flow()
    sys.exit(0 if success else 1)
