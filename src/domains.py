# src/domains.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd

from .model import CourseOffering

@dataclass(frozen=True)
class OfferDomain:
    teacher_ids: List[int]
    room_ids: List[int]
    allowed_day_idxs: List[int]
    allowed_start_slots: List[int]
    # Si quieres fijar algo (labs) puedes forzar options únicas:
    fixed_assignment: Optional[dict] = None

def split_weekly_hours(hours: int) -> Tuple[int, int]:
    """
    Divide weekly_hours en dos sesiones (día1 y día2) manteniendo consecutividad.
    Ejemplos típicos:
      6 -> (3,3)
      5 -> (3,2)
      4 -> (2,2)
      3 -> (2,1)
    """
    if hours <= 1:
        return (hours, 0)
    a = (hours + 1) // 2
    b = hours // 2
    return (a, b)

def build_offer_domains(
    offers: List[CourseOffering],
    aulas_df: pd.DataFrame,
    pref_df: pd.DataFrame,
    turn_slots: Dict[str, List[int]],
    n_days: int,
    baseline_lab: Dict[str, dict],  # cod_curso -> {day_idx,start,len,room_id,teacher_id}
) -> Dict[int, OfferDomain]:
    """
    Devuelve un dominio por índice de offering (misma posición que offers).
    - Docentes permitidos: de preferencia_docente_curso.csv (permitido=1)
    - Aulas compatibles: labs -> solo labs; no labs -> no-labs (puedes refinar por tipo/capacidad)
    - Slots permitidos: según turno
    - Laboratorio: si hay baseline, se fija (no alterable)
    """
    domains: Dict[int, OfferDomain] = {}

    lab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == True, "room_id"].dropna().unique().tolist()
    nonlab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == False, "room_id"].dropna().unique().tolist()

    for idx, off in enumerate(offers):
        # docentes permitidos
        sub = pref_df[(pref_df["cod_curso"] == off.cod_curso) & (pref_df["permitido"] == 1)]
        teacher_ids = sub["teacher_id"].dropna().astype(int).unique().tolist()

        # fallback: si no hay preferencia, permite todos (no ideal, pero evita romper)
        if not teacher_ids:
            teacher_ids = pref_df["teacher_id"].dropna().astype(int).unique().tolist()

        # aulas compatibles
        room_ids = lab_rooms if off.is_lab else nonlab_rooms

        # días permitidos: Lun..Sáb (0..5). Puedes restringir por disponibilidad si la tienes.
        allowed_day_idxs = list(range(n_days))

        # slots permitidos por turno (y luego filtramos por longitud de la sesión)
        allowed_start_slots = turn_slots.get(off.turno, [])

        fixed = None
        if off.is_lab and off.cod_curso in baseline_lab:
            # Lab fijo/no alterable
            fixed = baseline_lab[off.cod_curso].copy()

        domains[idx] = OfferDomain(
            teacher_ids=teacher_ids,
            room_ids=room_ids,
            allowed_day_idxs=allowed_day_idxs,
            allowed_start_slots=allowed_start_slots,
            fixed_assignment=fixed
        )

    return domains
