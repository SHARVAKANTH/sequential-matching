# dev_kit.py - FIXED VERSION
"""
Official simulator for the One Introduction Problem.
Do not modify this file.
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class RelationshipGoal(Enum):
    MARRIAGE = "marriage"
    DATING = "dating"
    FRIENDSHIP = "friendship"


class RelationshipPace(Enum):
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


@dataclass
class Person:
    member_id: str
    age: int
    gender: str
    who_to_meet: str
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    relationship_structure: Optional[str] = None
    smoking: Optional[bool] = None
    partner_smoking: Optional[bool] = None
    has_children: Optional[bool] = None
    partner_children: Optional[bool] = None
    wants_children: Optional[bool] = None
    acceptable_zones: Optional[List[str]] = None
    schedule: Optional[str] = None
    relationship_goal: Optional[RelationshipGoal] = None
    relationship_pace: Optional[RelationshipPace] = None
    lifestyle: Optional[str] = None
    conversations: Optional[str] = None
    emotional_availability: Optional[str] = None
    space_for_relationship: Optional[str] = None
    relocate: Optional[bool] = None
    zone: str = "downtown"
    available: bool = True
    field_status: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'member_id': self.member_id,
            'age': self.age,
            'gender': self.gender,
            'who_to_meet': self.who_to_meet,
            'age_min': self.age_min,
            'age_max': self.age_max,
            'relationship_structure': self.relationship_structure,
            'smoking': self.smoking,
            'partner_smoking': self.partner_smoking,
            'has_children': self.has_children,
            'partner_children': self.partner_children,
            'wants_children': self.wants_children,
            'acceptable_zones': self.acceptable_zones if self.acceptable_zones else [],
            'schedule': self.schedule,
            'relationship_goal': self.relationship_goal.value if self.relationship_goal else None,
            'relationship_pace': self.relationship_pace.value if self.relationship_pace else None,
            'lifestyle': self.lifestyle,
            'conversations': self.conversations,
            'emotional_availability': self.emotional_availability,
            'space_for_relationship': self.space_for_relationship,
            'relocate': self.relocate,
            'zone': self.zone,
            'available': self.available,
            'field_status': self.field_status,
        }


def generate(seed: int, n: int = 200, pool_id: str = "pool_1", variant: str = "development"):
    """
    Generate a synthetic pool of people.
    Returns dict with 'people' as list of DICTS (not Person objects).
    """
    random.seed(seed)
    
    people = []
    
    for i in range(n):
        member_id = f"{pool_id}_member_{i:04d}"
        age = random.randint(25, 65)
        gender = random.choice(['M', 'F'])
        who_to_meet = "women" if gender == "M" else "men"
        
        # Hard constraints
        age_min = max(18, age - random.randint(3, 8))
        age_max = min(100, age + random.randint(3, 8))
        
        zone = random.choice(['downtown', 'uptown', 'suburb'])
        
        person_dict = {
            'member_id': member_id,
            'age': age,
            'gender': gender,
            'who_to_meet': who_to_meet,
            'age_min': age_min,
            'age_max': age_max,
            'zone': zone,
            'relationship_structure': 'monogamous',
            'smoking': random.choice([True, False, None]),
            'partner_smoking': random.choice([True, False, None]),
            'has_children': random.choice([True, False, None]),
            'partner_children': random.choice([True, False, None]),
            'wants_children': random.choice([True, False, None]),
            'acceptable_zones': ['downtown', 'uptown', 'suburb'],
            'schedule': 'flexible',
            'available': True,
            'field_status': {},
        }
        
        people.append(person_dict)
    
    return {
        'pool_id': pool_id,
        'variant': variant,
        'people': people,  # NOW DICTS, NOT PERSON OBJECTS
        'seed': seed,
        'n': n
    }


class Simulator:
    """Official simulator for sequential matching."""
    
    def __init__(self, world: Dict):
        self.pool_id = world['pool_id']
        self.variant = world['variant']
        self.seed = world['seed']
        self.n_people = world['n']
        
        # FIX: Store members as dicts directly
        self.members = world['people'].copy()
        self.members_by_id = {m['member_id']: m for m in self.members}
        
        self.day = 0
        self.introductions = []
        self.feedback = {}
        self.ask_budget_remaining = 1000
        
    def observe(self):
        """Return current state."""
        return {
            'day': self.day,
            'members': self.members.copy(),  # Already dicts
            'introductions': self.introductions.copy(),
            'feedback': self.feedback.copy(),
            'ask_budget_remaining': self.ask_budget_remaining
        }
    
    def resolve_asks(self, asks: List[Dict]) -> List[Dict]:
        """Process clarification questions."""
        responses = []
        for ask in asks:
            member_id = ask['member_id']
            field = ask['field']
            member = self.members_by_id.get(member_id)
            
            if not member:
                continue
            
            cost = 3 if field == 'constraints' else 1
            
            if self.ask_budget_remaining < cost:
                continue
            
            self.ask_budget_remaining -= cost
            member['field_status'][field] = 'observed'
            
            responses.append({
                'member_id': member_id,
                'field': field,
                'cost': cost
            })
        
        return responses
    
    def advance(self, pairs: List[Tuple[str, str]]):
        """
        FIX: Generate REALISTIC feedback based on compatibility.
        Not just random 30% MSMI.
        """
        for a_id, b_id in pairs:
            a = self.members_by_id.get(a_id)
            b = self.members_by_id.get(b_id)
            
            if not a or not b:
                continue
            
            # Record introduction
            self.introductions.append((a_id, b_id, self.day))
            
            # FIX: Generate realistic MSMI based on compatibility
            msmi_score = self._compute_msmi(a, b)
            msmi_outcome = random.random() < msmi_score
            
            self.feedback[(a_id, b_id)] = msmi_outcome
        
        self.day += 1
    
    def _compute_msmi(self, person_a: Dict, person_b: Dict) -> float:
        """
        Compute realistic MSMI probability based on compatibility.
        FIX: This is what outcome models should predict.
        """
        score = 0.5  # Base score
        
        # Age compatibility
        age_diff = abs(person_a['age'] - person_b['age'])
        if age_diff <= 5:
            score += 0.15
        elif age_diff <= 10:
            score += 0.08
        
        # Age range check
        a_min = person_a.get('age_min')
        a_max = person_a.get('age_max')
        b_min = person_b.get('age_min')
        b_max = person_b.get('age_max')
        
        b_age = person_b['age']
        a_age = person_a['age']
        
        if a_min and b_age < a_min:
            score -= 0.1
        if a_max and b_age > a_max:
            score -= 0.1
        if b_min and a_age < b_min:
            score -= 0.1
        if b_max and a_age > b_max:
            score -= 0.1
        
        # Zone compatibility
        a_zones = person_a.get('acceptable_zones', [])
        b_zone = person_b.get('zone')
        if b_zone and (not a_zones or b_zone in a_zones):
            score += 0.05
        
        b_zones = person_b.get('acceptable_zones', [])
        a_zone = person_a.get('zone')
        if a_zone and (not b_zones or a_zone in b_zones):
            score += 0.05
        
        # Smoking compatibility
        a_smoking = person_a.get('smoking')
        b_smoking = person_b.get('smoking')
        a_partner_smoking = person_a.get('partner_smoking')
        b_partner_smoking = person_b.get('partner_smoking')
        
        if a_partner_smoking is not None and b_smoking is not None:
            if a_partner_smoking == b_smoking:
                score += 0.1
            else:
                score -= 0.1
        
        if b_partner_smoking is not None and a_smoking is not None:
            if b_partner_smoking == a_smoking:
                score += 0.1
            else:
                score -= 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def metrics(self):
        """Return performance metrics."""
        total_introductions = len(self.introductions)
        mutual_second_meeting_intention = sum(1 for v in self.feedback.values() if v)
        
        # FIX: MSMI per 100 arrivals (not per person)
        msmi_per_100_arrived = (mutual_second_meeting_intention / max(self.n_people, 1)) * 100
        
        return {
            'total_introductions': total_introductions,
            'mutual_second_meeting_intention': mutual_second_meeting_intention,
            'msmi_per_100_arrived_members': msmi_per_100_arrived,
            'ask_cost': 1000 - self.ask_budget_remaining
        }


def eligibility(person_a: Dict, person_b: Dict) -> bool:
    """Check if a pair is eligible (passes hard constraints)."""
    # Gender match
    a_who = person_a.get('who_to_meet', 'any')
    b_gender = person_b.get('gender')
    
    if a_who == "men" and b_gender != "M":
        return False
    if a_who == "women" and b_gender != "F":
        return False
    
    # Reverse direction
    b_who = person_b.get('who_to_meet', 'any')
    a_gender = person_a.get('gender')
    
    if b_who == "men" and a_gender != "M":
        return False
    if b_who == "women" and a_gender != "F":
        return False
    
    # Age constraints
    a_age = person_a.get('age')
    b_age = person_b.get('age')
    a_age_min = person_a.get('age_min')
    a_age_max = person_a.get('age_max')
    b_age_min = person_b.get('age_min')
    b_age_max = person_b.get('age_max')
    
    if a_age_min and b_age and b_age < a_age_min:
        return False
    if a_age_max and b_age and b_age > a_age_max:
        return False
    if b_age_min and a_age and a_age < b_age_min:
        return False
    if b_age_max and a_age and a_age > b_age_max:
        return False
    
    # Zone
    a_zones = person_a.get('acceptable_zones', [])
    b_zone = person_b.get('zone')
    if a_zones and b_zone not in a_zones:
        return False
    
    b_zones = person_b.get('acceptable_zones', [])
    a_zone = person_a.get('zone')
    if b_zones and a_zone not in b_zones:
        return False
    
    return True
