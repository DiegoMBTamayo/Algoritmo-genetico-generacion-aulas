# src/model.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

DayIdx = int
SlotIdx = int

@dataclass(frozen=True)
class CourseOffering:
    cod_curso: str
    nombre: str
    ciclo: int
    turno: str          # "M", "T", "N"
    grupo_horario: str  # ej. "01S"
    tipo_hora: str      # "T", "P", "L"
    weekly_hours: int   # 3/4/5/6 (o el que definas)
    is_lab: bool

@dataclass
class Gene:
    # Un “gen” = asignación para 1 offering (docente + 1-2 sesiones)
    teacher_id: int
    days: Tuple[DayIdx, ...]          # 1 o 2 días
    room1: int
    start1: SlotIdx
    len1: int
    room2: Optional[int] = None
    start2: Optional[SlotIdx] = None
    len2: Optional[int] = None
    frozen: bool = False              # para labs (no alterable)

@dataclass
class Individual:
    genes: List[Gene]
    penalty: float = 0.0
    fitness: float = 0.0
    scaled_fitness: float = 0.0
