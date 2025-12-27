# src/initial_population.py
from typing import Dict, List
import random

from .model import CourseOffering, Gene, Individual
from .domains import OfferDomain, split_weekly_hours

def _choose_two_distinct_days(day_pool: List[int]) -> tuple[int, int]:
    if len(day_pool) == 1:
        return day_pool[0], day_pool[0]
    d1 = random.choice(day_pool)
    remaining = [d for d in day_pool if d != d1] or [d1]
    d2 = random.choice(remaining)
    return d1, d2

def _valid_start_slots(allowed_starts: List[int], session_len: int, max_slot: int) -> List[int]:
    # start válido si start+len-1 <= max_slot
    out = []
    for s in allowed_starts:
        if s + session_len - 1 <= max_slot:
            out.append(s)
    return out

def random_gene_for_offer(
    off: CourseOffering,
    dom: OfferDomain,
    n_slots_per_day: int,
) -> Gene:
    # si es fijo (lab baseline), devolver asignación congelada
    if dom.fixed_assignment is not None:
        fa = dom.fixed_assignment
        # si baseline solo trae 1 sesión, congelamos 1
        return Gene(
            teacher_id=int(fa["teacher_id"]),
            days=(int(fa["day_idx"]),),
            room1=int(fa["room_id"]),
            start1=int(fa["start"]),
            len1=int(fa["length"]),
            frozen=True
        )

    teacher_id = random.choice(dom.teacher_ids)

    h1, h2 = split_weekly_hours(off.weekly_hours)
    max_slot = n_slots_per_day - 1

    # si solo una sesión
    if h2 == 0:
        day = random.choice(dom.allowed_day_idxs)
        starts = _valid_start_slots(dom.allowed_start_slots or list(range(max_slot + 1)), h1, max_slot)
        if not starts:
            starts = [0]
        start = random.choice(starts)
        room = random.choice(dom.room_ids)
        return Gene(
            teacher_id=teacher_id,
            days=(day,),
            room1=room,
            start1=start,
            len1=h1
        )

    # dos sesiones (día1/día2)
    d1, d2 = _choose_two_distinct_days(dom.allowed_day_idxs)

    starts1 = _valid_start_slots(dom.allowed_start_slots or list(range(max_slot + 1)), h1, max_slot)
    if not starts1:
        starts1 = [0]
    starts2 = _valid_start_slots(dom.allowed_start_slots or list(range(max_slot + 1)), h2, max_slot)
    if not starts2:
        starts2 = [0]

    start1 = random.choice(starts1)
    start2 = random.choice(starts2)

    room1 = random.choice(dom.room_ids)
    room2 = random.choice(dom.room_ids)

    return Gene(
        teacher_id=teacher_id,
        days=(d1, d2),
        room1=room1,
        start1=start1,
        len1=h1,
        room2=room2,
        start2=start2,
        len2=h2
    )

def build_random_individual(
    offers: List[CourseOffering],
    domains: Dict[int, OfferDomain],
    n_slots_per_day: int
) -> Individual:
    genes: List[Gene] = []
    for i, off in enumerate(offers):
        g = random_gene_for_offer(off, domains[i], n_slots_per_day)
        genes.append(g)
    return Individual(genes=genes)

def build_initial_population(
    offers: List[CourseOffering],
    domains: Dict[int, OfferDomain],
    n_slots_per_day: int,
    pop_size: int
) -> List[Individual]:
    return [build_random_individual(offers, domains, n_slots_per_day) for _ in range(pop_size)]
