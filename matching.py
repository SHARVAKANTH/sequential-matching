# matching.py
"""
Matching of members into non-overlapping pairs.

Two strategies:
1. greedy_max_weight_matching: always match, highest scores first
2. matching_with_waiting_threshold: only match pairs scoring >= threshold,
   and wait (introduce nobody) if nothing qualifies

Both NEVER select a pair with score <= 0. Infeasible pairs (hard-constraint
violations) are given score 0 upstream, so skipping them here is what keeps
the hard rules unbreakable even when the matcher has no threshold.
"""


def greedy_max_weight_matching(scores, members_by_id=None):
    """
    Greedy approximation to maximum-weight matching.

    Args:
        scores: {(a_id, b_id): score in [0, 1]}
        members_by_id: unused, kept for API compatibility

    Returns:
        list of (a_id, b_id)
    """
    if not scores:
        return []

    matched = set()
    pairs = []
    for (a_id, b_id), score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score <= 0:
            break  # sorted descending: everything after is infeasible too
        if a_id not in matched and b_id not in matched:
            pairs.append((a_id, b_id))
            matched.add(a_id)
            matched.add(b_id)
    return pairs


def matching_with_waiting_threshold(scores, members_by_id=None, threshold=0.5):
    """
    Match only pairs with score >= threshold; wait if none qualify.

    Returns:
        (pairs, should_wait, total_score)
    """
    if not scores:
        return [], True, 0.0

    cutoff = max(threshold, 1e-12)  # never accept score 0, even with threshold 0
    matched = set()
    pairs = []
    total = 0.0
    for (a_id, b_id), score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score < cutoff:
            break
        if a_id not in matched and b_id not in matched:
            pairs.append((a_id, b_id))
            matched.add(a_id)
            matched.add(b_id)
            total += score

    return pairs, len(pairs) == 0, total
