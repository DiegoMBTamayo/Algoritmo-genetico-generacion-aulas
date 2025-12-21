# src/evaluation.py
from typing import List
from .model import Individual, CourseOffering
from .config import GAConfig

def evaluate(
    ind: Individual,
    offers: List[CourseOffering],
    n_cycles: int,
    n_rooms: int,
    n_teachers: int,
    cfg: GAConfig
) -> float:
    # matrices: [x][day][slot]
    ch_cycle = [[[-1]*cfg.N_SLOTS_PER_DAY for _ in range(cfg.N_DAYS)] for _ in range(n_cycles)]
    ch_room  = [[[-1]*cfg.N_SLOTS_PER_DAY for _ in range(cfg.N_DAYS)] for _ in range(n_rooms)]
    ch_teach = [[[-1]*cfg.N_SLOTS_PER_DAY for _ in range(cfg.N_DAYS)] for _ in range(n_teachers + 1)]

    pen = 0

    for i, (g, off) in enumerate(zip(ind.genes, offers)):
        pen += _occupy(off, i, g.teacher_id, g.room1, g.days[0], g.start1, g.len1,
                       ch_cycle, ch_room, ch_teach, cfg)

        if len(g.days) > 1 and g.room2 is not None and g.start2 is not None and g.len2 is not None:
            pen += _occupy(off, i, g.teacher_id, g.room2, g.days[1], g.start2, g.len2,
                           ch_cycle, ch_room, ch_teach, cfg)

    ind.penalty = pen
    ind.fitness = 1.0 / (1.0 + pen + cfg.EPS)
    return pen

def _occupy(off, course_idx, teacher_id, room_id, day, start, length,
            ch_cycle, ch_room, ch_teach, cfg: GAConfig) -> int:
    p = 0
    cyc = int(off.ciclo) - 1
    for s in range(start, start + length):
        # choque por ciclo-hora
        if ch_cycle[cyc][day][s] != -1:
            p += cfg.W_CONFLICT_CYCLE
        else:
            ch_cycle[cyc][day][s] = course_idx

        # choque aula
        if ch_room[room_id][day][s] != -1:
            p += cfg.W_CONFLICT_ROOM
        else:
            ch_room[room_id][day][s] = course_idx

        # choque docente
        if ch_teach[teacher_id][day][s] != -1:
            p += cfg.W_CONFLICT_TEACHER
        else:
            ch_teach[teacher_id][day][s] = course_idx
    return p
