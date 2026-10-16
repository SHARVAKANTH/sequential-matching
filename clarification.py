# clarification.py - FIXED VERSION
"""
Value-of-information based clarification.

Selects which questions to ask to reduce uncertainty and improve matching.
"""

import random


class ValueOfInformationSelector:
    """Select which questions to ask based on expected value."""
    
    def __init__(self, ensemble, seed=42):
        """
        Initialize VOI selector.
        
        Args:
            ensemble: OutcomeEnsemble instance
            seed: Random seed
        """
        self.ensemble = ensemble
        self.seed = seed
        self.day = 0
    
    def estimate_voi(self, member, field):
        """
        Estimate value of information for asking member about field.
        
        Simple heuristic: constraint fields have higher value.
        
        Args:
            member: Member dict
            field: Field name (string)
        
        Returns:
            float: estimated value
        """
        # Constraint fields are high-value (unlock feasibility)
        if field == 'constraints':
            return 2.0
        
        # Soft fields are lower-value but help with scoring
        if field in ['relationship_goal', 'lifestyle', 'conversations',
                     'emotional_availability', 'space_for_relationship']:
            return 1.0
        
        # Unknown field
        return 0.5
    
    def select_asks(self, state, max_cost=12):
        """
        Greedily select questions up to budget using VOI heuristic.
        
        Algorithm:
        1. For each member, find unobserved fields
        2. Estimate VOI for each (value / cost ratio)
        3. Greedily select highest-value asks until budget exhausted
        
        Args:
            state: Simulator state dict with 'members'
            max_cost: Maximum total ask cost (default: 12)
        
        Returns:
            list of {'member_id': ..., 'field': ...} dicts
        """
        # FIX: Safety check
        if not state or 'members' not in state:
            return []
        
        members = state['members']
        if not members:
            return []
        
        candidates = []
        
        for member in members:
            # FIX: Check if member is available
            if not member.get('available', True):
                continue
            
            member_id = member.get('member_id')
            if not member_id:
                continue
            
            # FIX: Safe access to field_status
            field_status = member.get('field_status', {})
            if not isinstance(field_status, dict):
                field_status = {}
            
            # --- CONSTRAINT FIELDS (cost=3) ---
            # Ask about constraints if not yet observed
            if field_status.get('constraints') != 'observed':
                voi = self.estimate_voi(member, 'constraints')
                cost = 3
                
                # FIX: Only add if VOI > 0
                if voi > 0:
                    candidates.append({
                        'member_id': member_id,
                        'field': 'constraints',
                        'voi': voi,
                        'cost': cost,
                        'voi_per_cost': voi / cost if cost > 0 else 0
                    })
            
            # --- SOFT FIELDS (cost=1) ---
            soft_fields = [
                'relationship_goal',
                'lifestyle',
                'conversations',
                'emotional_availability',
                'space_for_relationship'
            ]
            
            for field in soft_fields:
                if field_status.get(field) != 'observed':
                    voi = self.estimate_voi(member, field)
                    cost = 1
                    
                    # FIX: Only add if VOI > 0
                    if voi > 0:
                        candidates.append({
                            'member_id': member_id,
                            'field': field,
                            'voi': voi,
                            'cost': cost,
                            'voi_per_cost': voi / cost if cost > 0 else 0
                        })
        
        if not candidates:
            return []
        
        # FIX: Sort by value-to-cost ratio (descending)
        try:
            candidates.sort(key=lambda x: x.get('voi_per_cost', 0), reverse=True)
        except Exception as e:
            print(f"ERROR sorting candidates: {e}")
            return []
        
        # Greedily select up to budget
        total_cost = 0
        asks = []
        
        for candidate in candidates:
            cost = candidate.get('cost', 0)
            
            # FIX: Check if adding this candidate exceeds budget
            if total_cost + cost <= max_cost:
                asks.append({
                    'member_id': candidate['member_id'],
                    'field': candidate['field']
                })
                total_cost += cost
        
        return asks
    
    def advance_day(self):
        """Move to next day."""
        self.day += 1
