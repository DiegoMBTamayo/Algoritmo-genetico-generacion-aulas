import copy
import random
from typing import Dict, List, Tuple
from .model import Individual, Gene, CourseOffering
from .domains import split_weekly_hours, OfferDomain
from .config import GAConfig


def _valid_starts(allowed_starts: List[int], length: int, max_slot: int) -> List[int]:
    return [s for s in allowed_starts if s + length <= max_slot]


def repair_gene(gene: Gene, off: CourseOffering, dom: OfferDomain, cfg: GAConfig) -> Gene:
    """Ajusta un gen para que cumpla disponibilidad, turnos y tipos de aula."""
    h1, h2 = split_weekly_hours(off.weekly_hours)
    gene.len1 = h1
    gene.len2 = h2 if h2 > 0 else None

    # Docente permitido
    if gene.teacher_id not in dom.teacher_ids:
        gene.teacher_id = random.choice(dom.teacher_ids)

    # Días válidos
    day_pool = dom.allowed_day_idxs if dom.allowed_day_idxs else list(range(cfg.n_days))
    if h2 == 0:
        d1 = gene.days[0] if len(gene.days) > 0 and gene.days[0] in day_pool else random.choice(day_pool)
        gene.days = (d1,)
    else:
        d1 = gene.days[0] if len(gene.days) > 0 and gene.days[0] in day_pool else random.choice(day_pool)
        rest = [d for d in day_pool if d != d1] or [d1]
        d2 = gene.days[1] if len(gene.days) > 1 and gene.days[1] in rest else random.choice(rest)
        gene.days = (d1, d2)

    # Aulas válidas (ROE2/ROE3)
    room_pool = dom.room_ids if dom.room_ids else [gene.room1 or 0]
    if gene.room1 not in room_pool:
        gene.room1 = random.choice(room_pool)
    if h2 > 0:
        if gene.room2 not in room_pool:
            gene.room2 = random.choice(room_pool)

    # Horas válidas dentro de los slots del turno
    starts1 = _valid_starts(dom.allowed_start_slots or list(range(cfg.n_slots_per_day)), h1, cfg.n_slots_per_day)
    if not starts1:
        starts1 = [0]
    if gene.start1 not in starts1:
        gene.start1 = random.choice(starts1)

    if h2 and h2 > 0:
        starts2 = _valid_starts(dom.allowed_start_slots or list(range(cfg.n_slots_per_day)), h2, cfg.n_slots_per_day)
        if not starts2:
            starts2 = [0]
        if gene.start2 not in starts2:
            gene.start2 = random.choice(starts2)
    else:
        gene.start2 = None
        gene.room2 = None

    # Asignaciones fijas
    if dom.fixed_assignment:
        fa = dom.fixed_assignment
        if "teacher_id" in fa:
            gene.teacher_id = int(fa["teacher_id"])
        if "room_id" in fa:
            gene.room1 = int(fa["room_id"])
            gene.room2 = int(fa["room_id"]) if h2 else None
        if "day_idx" in fa:
            gene.days = (int(fa["day_idx"]),) if h2 == 0 else (int(fa["day_idx"]), gene.days[-1])
        if "start" in fa:
            gene.start1 = int(fa["start"])
        if "length" in fa:
            gene.len1 = int(fa["length"])
            gene.len2 = None
        gene.frozen = True

    return gene


def repair_individual(
    ind: Individual,
    offers: List[CourseOffering],
    domains: Dict[int, OfferDomain],
    cfg: GAConfig,
) -> Individual:
    for idx, (g, off) in enumerate(zip(ind.genes, offers)):
        if g.frozen:
            continue
        fixed = repair_gene(g, off, domains[idx], cfg)
        ind.genes[idx] = fixed
    return ind


def interleaved_crossover(p1: Individual, p2: Individual) -> Individual:
    """Cruce por campos intercalados respetando el orden días→aulas→horas."""
    child_genes: List[Gene] = []
    for g1, g2 in zip(p1.genes, p2.genes):
        better = g1 if p1.fitness >= p2.fitness else g2
        take_first = True
        days = g2.days if take_first else g1.days
        take_first = not take_first
        room1 = g1.room1 if take_first else g2.room1
        take_first = not take_first
        start1 = g2.start1 if take_first else g1.start1
        take_first = not take_first
        room2 = g1.room2 if take_first else g2.room2
        take_first = not take_first
        start2 = g2.start2 if take_first else g1.start2

        child_gene = Gene(
            teacher_id=better.teacher_id,
            days=days,
            room1=room1,
            start1=start1,
            len1=better.len1,
            room2=room2,
            start2=start2,
            len2=better.len2,
            frozen=g1.frozen or g2.frozen,
        )
        child_genes.append(child_gene)
    return Individual(genes=child_genes)


def mutate_swap(ind: Individual, mutation_rate: float):
    """Mutación por intercambio: intercambia bloques 1↔2 si existen."""
    for g in ind.genes:
        if g.frozen or g.len2 in (None, 0):
            continue
        if random.random() < mutation_rate:
            g.days = (g.days[1], g.days[0])
            g.room1, g.room2 = g.room2, g.room1
            g.start1, g.start2 = g.start2, g.start1
            g.len1, g.len2 = g.len2, g.len1


def sigma_scale(population: List[Individual], cfg: GAConfig) -> None:
    """Escalamiento sigma para evitar super-individuos."""
    raw = [ind.fitness for ind in population]
    mean = sum(raw) / len(raw)
    variance = sum((r - mean) ** 2 for r in raw) / len(raw)
    sigma = variance ** 0.5
    for ind, r in zip(population, raw):
        if sigma == 0:
            ind.scaled_fitness = r
        else:
            ind.scaled_fitness = max(r + (r - mean) / (cfg.fitness_sigma * sigma), 0.0)
