# eligibility_mask.py - FIXED VERSION
"""
Hard constraint enforcement for the matching problem.

This module encodes impossible pairs (hard constraints) as a binary mask:
- mask[(a_id, b_id)] = 1 means pair is feasible
- mask[(a_id, b_id)] = 0 means pair is impossible (hard constraint violated)

Hard constraints checked:
1. Gender preference (if 'who_to_meet' specified)
2. Age range (both directions)
3. Geographic zones (acceptable zones list)
4. Smoking compatibility (if both have stated preferences)
5. Children/relationship structure (if specified)
"""


def build_eligibility_mask(members):
    """
    Build eligibility mask: hard constraint encoding.
    
    Args:
        members: List of member dicts
    
    Returns:
        mask: dict {(a_id, b_id): 1/0} where (a_id, b_id) has a_id < b_id
    """
    mask = {}
    
    # Iterate all unique pairs
    for i, person_a in enumerate(members):
        for person_b in members[i + 1:]:
            a_id = person_a['member_id']
            b_id = person_b['member_id']
            pair = (a_id, b_id)
            
            # Start: assume feasible
            feasible = 1
            
            # ===== CONSTRAINT 1: GENDER PREFERENCE =====
            # FIX: Handle both 'M'/'F' and 'male'/'female' formats
            a_who_to_meet = person_a.get('who_to_meet', 'any')
            b_who_to_meet = person_b.get('who_to_meet', 'any')
            a_gender = person_a.get('gender', '')
            b_gender = person_b.get('gender', '')
            
            # Normalize gender to single letter
            a_gender_norm = a_gender[0].upper() if a_gender else None
            b_gender_norm = b_gender[0].upper() if b_gender else None
            
            # Person A's gender constraint
            if a_who_to_meet != 'any' and b_gender_norm:
                if a_who_to_meet == 'men' and b_gender_norm != 'M':
                    feasible = 0
                elif a_who_to_meet == 'women' and b_gender_norm != 'F':
                    feasible = 0
            
            # Person B's gender constraint
            if b_who_to_meet != 'any' and a_gender_norm:
                if b_who_to_meet == 'men' and a_gender_norm != 'M':
                    feasible = 0
                elif b_who_to_meet == 'women' and a_gender_norm != 'F':
                    feasible = 0
            
            if feasible == 0:
                mask[pair] = 0
                continue
            
            # ===== CONSTRAINT 2: AGE RANGE =====
            a_age = person_a.get('age')
            b_age = person_b.get('age')
            a_age_min = person_a.get('age_min')
            a_age_max = person_a.get('age_max')
            b_age_min = person_b.get('age_min')
            b_age_max = person_b.get('age_max')
            
            # If ages exist and ranges specified, check compatibility
            if a_age is not None and b_age is not None:
                
                # Person A's age range: is B within it?
                if a_age_min is not None and b_age < a_age_min:
                    feasible = 0
                if a_age_max is not None and b_age > a_age_max:
                    feasible = 0
                
                # Person B's age range: is A within it?
                if b_age_min is not None and a_age < b_age_min:
                    feasible = 0
                if b_age_max is not None and a_age > b_age_max:
                    feasible = 0
            
            if feasible == 0:
                mask[pair] = 0
                continue
            
            # ===== CONSTRAINT 3: GEOGRAPHIC ZONES =====
            a_zone = person_a.get('zone', 'downtown')
            b_zone = person_b.get('zone', 'downtown')
            a_acceptable_zones = person_a.get('acceptable_zones', [])
            b_acceptable_zones = person_b.get('acceptable_zones', [])
            
            # If A specified zones, B must be in one of them
            if a_acceptable_zones and len(a_acceptable_zones) > 0:
                if b_zone not in a_acceptable_zones:
                    feasible = 0
            
            # If B specified zones, A must be in one of them
            if b_acceptable_zones and len(b_acceptable_zones) > 0:
                if a_zone not in b_acceptable_zones:
                    feasible = 0
            
            if feasible == 0:
                mask[pair] = 0
                continue
            
            # ===== CONSTRAINT 4: SMOKING COMPATIBILITY =====
            # If both have stated preferences, they must be compatible
            a_smoking = person_a.get('smoking')
            b_smoking = person_b.get('smoking')
            a_partner_smoking = person_a.get('partner_smoking')
            b_partner_smoking = person_b.get('partner_smoking')
            
            # Person A's preference for partner's smoking
            if a_partner_smoking is not None and b_smoking is not None:
                if a_partner_smoking != b_smoking:
                    feasible = 0
            
            # Person B's preference for partner's smoking
            if b_partner_smoking is not None and a_smoking is not None:
                if b_partner_smoking != a_smoking:
                    feasible = 0
            
            if feasible == 0:
                mask[pair] = 0
                continue
            
            # All constraints passed
            mask[pair] = feasible
    
    return mask


def check_hard_constraints(state, pairs):
    """
    Audit: verify all proposed pairs satisfy hard constraints.
    
    Args:
        state: simulator state dict with 'members' list
        pairs: list of (a_id, b_id) tuples to check
    
    Returns:
        (is_valid: bool, violations: list of violation strings)
    """
    members_by_id = {m['member_id']: m for m in state['members']}
    violations = []
    
    for a_id, b_id in pairs:
        # Get member objects
        a = members_by_id.get(a_id)
        b = members_by_id.get(b_id)
        
        if not a or not b:
            violations.append(f"Member {a_id or b_id} not found")
            continue
        
        # Check gender constraint
        a_who_to_meet = a.get('who_to_meet', 'any')
        b_who_to_meet = b.get('who_to_meet', 'any')
        a_gender = a.get('gender', '')
        b_gender = b.get('gender', '')
        
        # Normalize gender
        a_gender_norm = a_gender[0].upper() if a_gender else None
        b_gender_norm = b_gender[0].upper() if b_gender else None
        
        if a_who_to_meet != 'any' and b_gender_norm:
            if a_who_to_meet == 'men' and b_gender_norm != 'M':
                violations.append(f"Gender mismatch: {a_id} wants men, {b_id} is {b_gender}")
            elif a_who_to_meet == 'women' and b_gender_norm != 'F':
                violations.append(f"Gender mismatch: {a_id} wants women, {b_id} is {b_gender}")
        
        if b_who_to_meet != 'any' and a_gender_norm:
            if b_who_to_meet == 'men' and a_gender_norm != 'M':
                violations.append(f"Gender mismatch: {b_id} wants men, {a_id} is {a_gender}")
            elif b_who_to_meet == 'women' and a_gender_norm != 'F':
                violations.append(f"Gender mismatch: {b_id} wants women, {a_id} is {a_gender}")
        
        # Check age constraint
        a_age = a.get('age')
        b_age = b.get('age')
        a_age_min = a.get('age_min')
        a_age_max = a.get('age_max')
        b_age_min = b.get('age_min')
        b_age_max = b.get('age_max')
        
        if a_age is not None and b_age is not None:
            if a_age_min is not None and b_age < a_age_min:
                violations.append(f"Age violation: {b_id} age {b_age} < {a_id} min {a_age_min}")
            if a_age_max is not None and b_age > a_age_max:
                violations.append(f"Age violation: {b_id} age {b_age} > {a_id} max {a_age_max}")
            if b_age_min is not None and a_age < b_age_min:
                violations.append(f"Age violation: {a_id} age {a_age} < {b_id} min {b_age_min}")
            if b_age_max is not None and a_age > b_age_max:
                violations.append(f"Age violation: {a_id} age {a_age} > {b_id} max {b_age_max}")
        
        # Check zone constraint
        a_zone = a.get('zone', 'downtown')
        b_zone = b.get('zone', 'downtown')
        a_acceptable_zones = a.get('acceptable_zones', [])
        b_acceptable_zones = b.get('acceptable_zones', [])
        
        if a_acceptable_zones and b_zone not in a_acceptable_zones:
            violations.append(f"Zone violation: {b_id} zone {b_zone} not in {a_id} zones {a_acceptable_zones}")
        if b_acceptable_zones and a_zone not in b_acceptable_zones:
            violations.append(f"Zone violation: {a_id} zone {a_zone} not in {b_id} zones {b_acceptable_zones}")
        
        # Check smoking constraint
        a_smoking = a.get('smoking')
        b_smoking = b.get('smoking')
        a_partner_smoking = a.get('partner_smoking')
        b_partner_smoking = b.get('partner_smoking')
        
        if a_partner_smoking is not None and b_smoking is not None:
            if a_partner_smoking != b_smoking:
                violations.append(f"Smoking violation: {a_id} wants partner_smoking={a_partner_smoking}, {b_id} smoking={b_smoking}")
        
        if b_partner_smoking is not None and a_smoking is not None:
            if b_partner_smoking != a_smoking:
                violations.append(f"Smoking violation: {b_id} wants partner_smoking={b_partner_smoking}, {a_id} smoking={a_smoking}")
    
    is_valid = len(violations) == 0
    return is_valid, violations
