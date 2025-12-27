# src/evaluation.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict
import numpy as np
from .model import Individual, CourseOffering
from .config import GAConfig


@dataclass
class EvaluationResult:
    penalty: int
    hi: int
    ai: int
    pi: int
    conflict_cycle: np.ndarray
    conflict_room: np.ndarray
    conflict_teacher: np.ndarray
    teacher_hours: Dict[int, int]
    violations: List[str]


def evaluate(
    ind: Individual,
    offers: List[CourseOffering],
    n_cycles: int,
    n_rooms: int,
    n_teachers: int,
    cfg: GAConfig,
) -> EvaluationResult:
    # Matrices [entity][day][slot]
    ch_cycle = np.zeros((n_cycles, cfg.n_days, cfg.n_slots_per_day), dtype=int)
    ch_room = np.zeros((n_rooms, cfg.n_days, cfg.n_slots_per_day), dtype=int)
    ch_teach = np.zeros((n_teachers + 1, cfg.n_days, cfg.n_slots_per_day), dtype=int)

    hi = ai = pi = 0
    teacher_hours: Dict[int, int] = {}
    violations: List[str] = []
    group_occ: DefaultDict[Tuple[int, str], np.ndarray] = defaultdict(
        lambda: np.zeros((cfg.n_days, cfg.n_slots_per_day), dtype=int)
    )

    for g, off in zip(ind.genes, offers):
        blocks = [
            (g.days[0] if len(g.days) > 0 else None, g.start1, g.len1, g.room1),
            (g.days[1] if len(g.days) > 1 else None, g.start2, g.len2, g.room2),
        ]
        for day, start, length, room in blocks:
            if day is None or start is None or length in (None, 0):
                continue
            cyc = max(0, int(off.ciclo) - 1)
            for s in range(start, start + length):
                if s >= cfg.n_slots_per_day or day >= cfg.n_days:
                    continue
                if s not in cfg.turn_slots.get(off.turno, []):
                    violations.append(f"Turno inválido {off.turno} en slot {s} para {off.cod_curso}")
                    pi += 1  # penalizamos con el peso de docente
                ch_cycle[cyc, day, s] += 1
                ch_room[room, day, s] += 1
                ch_teach[g.teacher_id, day, s] += 1
                key = (cyc, off.grupo_horario)
                group_occ[key][day, s] += 1
                if ch_cycle[cyc, day, s] > 1:
                    hi += 1
                if ch_room[room, day, s] > 1:
                    ai += 1
                if ch_teach[g.teacher_id, day, s] > 1:
                    pi += 1
                if group_occ[key][day, s] > 1:
                    violations.append(f"Choque grupo {off.grupo_horario} ciclo {off.ciclo} en día {day} slot {s}")
            teacher_hours[g.teacher_id] = teacher_hours.get(g.teacher_id, 0) + length

    # ROE1: horas mín/max por docente
    for tid, hours in teacher_hours.items():
        limit = cfg.teacher_hour_limits.get(str(tid), cfg.teacher_hour_limits.get("default", {"min": 0, "max": 10}))
        if hours < limit["min"] or hours > limit["max"]:
            pi += 1
            violations.append(f"Horas fuera de rango para docente {tid}: {hours}")

    penalty = (
        cfg.weight_conflict_cycle * hi
        + cfg.weight_conflict_room * ai
        + cfg.weight_conflict_teacher * pi
    )
    ind.penalty = penalty
    ind.fitness = 1.0 / (1.0 + penalty + cfg.eps)

    return EvaluationResult(
        penalty=penalty,
        hi=hi,
        ai=ai,
        pi=pi,
        conflict_cycle=ch_cycle,
        conflict_room=ch_room,
        conflict_teacher=ch_teach,
        teacher_hours=teacher_hours,
        violations=violations,
    )
