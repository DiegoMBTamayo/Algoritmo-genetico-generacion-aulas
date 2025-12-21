# src/config.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class GAConfig:
    # Días: Lun..Sáb (6)
    N_DAYS: int = 6

    # Slots por día (ajústalo a tu grilla real)
    # En tu doc aparece 6 días y ~15 horas lectivas como ejemplo.
    N_SLOTS_PER_DAY: int = 15

    # Pesos de choques (doc: horas=5, aula=4, docente=2)
    W_CONFLICT_CYCLE: int = 5
    W_CONFLICT_ROOM: int = 4
    W_CONFLICT_TEACHER: int = 2

    # Fitness
    EPS: float = 1e-9

# Mapeo de turnos a slots permitidos (AJUSTA si tu grilla es distinta)
# La idea es: mañana = primeros slots; tarde = intermedios; noche = últimos.
TURN_SLOTS: Dict[str, List[int]] = {
    "M": list(range(0, 5)),    # 5 slots para mañana
    "T": list(range(5, 10)),   # 5 slots para tarde
    "N": list(range(10, 15)),  # 5 slots para noche
}

DAY_NAME_TO_IDX = {
    "LUNES": 0,
    "MARTES": 1,
    "MIÉRCOLES": 2,
    "MIERCOLES": 2,
    "JUEVES": 3,
    "VIERNES": 4,
    "SÁBADO": 5,
    "SABADO": 5,
}
