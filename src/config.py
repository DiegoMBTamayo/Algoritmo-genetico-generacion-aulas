"""
Configuración del algoritmo genético.

Incluye un cargador desde YAML (o JSON compatible) para dejar los
parámetros reproducibles y configurables tal como pide la especificación.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

try:  # PyYAML puede no estar instalado; usamos el parser si existe.
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_TURN_SLOTS: Dict[str, List[int]] = {
    # 17 periodos diarios (0..16). Turnos con solape tarde/noche.
    "M": list(range(0, 6)),     # 08:00-13:00
    "T": list(range(6, 12)),    # 13:00-18:00 (incluye el solape 17-18)
    "N": list(range(11, 17)),   # 17:00-22:10
}

DEFAULT_CYCLE_TURN_MAP: List[Tuple[Tuple[int, int], str]] = [
    ((1, 3), "M"),
    ((4, 7), "T"),
    ((8, 10), "N"),
]


@dataclass
class GAConfig:
    # Tiempo
    n_days: int = 6
    n_slots_per_day: int = 17
    turn_slots: Dict[str, List[int]] = field(default_factory=lambda: DEFAULT_TURN_SLOTS.copy())
    cycle_turn_map: List[Tuple[Tuple[int, int], str]] = field(
        default_factory=lambda: DEFAULT_CYCLE_TURN_MAP.copy()
    )

    # Algoritmo genético
    population_size: int = 102
    generations: int = 60
    mutation_rate: float = 0.1
    elite_size: int = 1
    max_stagnation: int = 20
    seed: int = 42

    # Pesos y fitness
    weight_conflict_cycle: int = 5
    weight_conflict_room: int = 4
    weight_conflict_teacher: int = 2
    eps: float = 1e-9
    fitness_sigma: float = 2.0  # para escalamiento de aptitud

    # Dominio
    weekly_hours_by_type: Dict[str, int] = field(
        default_factory=lambda: {"T": 3, "P": 2, "L": 2}
    )
    teacher_hour_limits: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"default": {"min": 0, "max": 16}}
    )
    course_teacher_pref: Dict[str, List[int]] = field(default_factory=dict)
    fixed_course_room: List[Dict[str, Any]] = field(default_factory=list)
    fixed_teacher_course: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAConfig":
        merged = asdict(cls())
        for k, v in data.items():
            if k in merged:
                merged[k] = v
        return cls(**merged)

    def __post_init__(self):
        # Alias legacy (app.py)
        self.N_DAYS = self.n_days
        self.N_SLOTS_PER_DAY = self.n_slots_per_day
        self.W_CONFLICT_CYCLE = self.weight_conflict_cycle
        self.W_CONFLICT_ROOM = self.weight_conflict_room
        self.W_CONFLICT_TEACHER = self.weight_conflict_teacher


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except Exception:
        # Si no hay parser disponible, ignoramos el archivo y usamos defaults.
        return {}


def load_config(path: str = "config.yaml") -> GAConfig:
    cfg_path = Path(path)
    data = _load_yaml_or_json(cfg_path)
    if not isinstance(data, dict):
        raise ValueError("config.yaml debe contener un objeto mapeo")
    return GAConfig.from_dict(data)


def resolve_turn_for_cycle(cycle: int, mapping: List[Tuple[Tuple[int, int], str]]) -> str:
    """
    Retorna el turno para un ciclo usando el mapeo configurable.
    """
    for (start, end), turn in mapping:
        if start <= cycle <= end:
            return turn
    return "M"

# Compatibilidad con código previo
TURN_SLOTS = DEFAULT_TURN_SLOTS
