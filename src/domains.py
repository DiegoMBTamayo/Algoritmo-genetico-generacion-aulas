# src/domains.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
import pandas as pd
from .model import CourseOffering
from .config import GAConfig


@dataclass(frozen=True)
class OfferDomain:
    teacher_ids: List[int]
    room_ids: List[int]
    allowed_day_idxs: List[int]
    allowed_start_slots: List[int]
    fixed_assignment: Optional[dict] = None
    turn: Optional[str] = None


def split_weekly_hours(hours: int) -> Tuple[int, int]:
    """
    Regla ROG6: si par -> 2/2..., si impar -> 3 + 2/2...
    Para mantener compatibilidad con el cromosoma (2 bloques máximo),
    devolvemos dos tramos. Para horas >4, concentramos las horas restantes
    en el segundo bloque (documentado en decisiones).
    """
    if hours <= 1:
        return (hours, 0)
    if hours == 2:
        return (2, 0)
    if hours == 3:
        return (3, 0)
    if hours == 4:
        return (2, 2)
    if hours == 5:
        return (3, 2)
    # 6 o más: concentramos en dos bloques consecutivos
    first = min(3, hours // 2 + hours % 2)
    second = hours - first
    return (first, second)


def get_allowed_turns(cod_dispo: int) -> Set[str]:
    """
    Mapea el código de disponibilidad (docentes.csv) a los turnos permitidos (M, T, N).
    1: Mañana, 2: Tarde, 3: Noche, 4: Mañana-Tarde, 5: Tarde-Noche, 6: Tiempo completo
    """
    mapping = {
        1: {"M"},
        2: {"T"},
        3: {"N"},
        4: {"M", "T"},
        5: {"T", "N"},
        6: {"M", "T", "N"}
    }
    return mapping.get(int(cod_dispo), set())


def _find_fixed_assignment(
    off: CourseOffering,
    fixed_course_room: List[Dict[str, Any]],
    fixed_teacher_course: List[Dict[str, Any]],
) -> Optional[dict]:
    key = (off.cod_curso, off.grupo_horario, off.tipo_hora)
    for rec in fixed_course_room:
        if (rec.get("cod_curso"), rec.get("grupo_horario"), rec.get("tipo_hora")) == key:
            return {
                "room_id": int(rec["room_id"]),
                "day_idx": int(rec["day_idx"]),
                "start": int(rec["start"]),
                "length": int(rec.get("length", off.weekly_hours)),
            }
    for rec in fixed_teacher_course:
        if (rec.get("cod_curso"), rec.get("grupo_horario"), rec.get("tipo_hora")) == key:
            return {
                "teacher_id": int(rec["teacher_id"]),
                "day_idx": int(rec["day_idx"]),
                "start": int(rec["start"]),
                "length": int(rec.get("length", off.weekly_hours)),
            }
    return None


def build_offer_domains(
    offers: List[CourseOffering],
    aulas_df: pd.DataFrame,
    docentes_df: pd.DataFrame,
    cfg: GAConfig,
    baseline_lab: Dict[str, dict],
) -> Dict[int, OfferDomain]:

    domains: Dict[int, OfferDomain] = {}

    lab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == True, "room_id"].dropna().unique().tolist()
    nonlab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == False, "room_id"].dropna().unique().tolist()

    teacher_availability = {}
    for _, row in docentes_df.iterrows():
        tid = int(str(row["cod_docente"]).replace("D", ""))
        cdispo = int(row.get("cod_dispo", 6)) if pd.notna(row.get("cod_dispo", 6)) else 6
        teacher_availability[tid] = get_allowed_turns(cdispo)

    all_teachers = list(teacher_availability.keys())

    for idx, off in enumerate(offers):
        # Disponibilidad
        pref_teachers = cfg.course_teacher_pref.get(off.cod_curso, all_teachers)
        valid_teachers = []
        for tid in pref_teachers:
            allowed = teacher_availability.get(tid, set())
            if off.turno in allowed:
                valid_teachers.append(tid)
        if not valid_teachers:
            valid_teachers = pref_teachers if pref_teachers else all_teachers

        room_ids = lab_rooms if off.is_lab else nonlab_rooms
        allowed_day_idxs = list(range(cfg.n_days))
        allowed_start_slots = cfg.turn_slots.get(off.turno, [])

        fixed = baseline_lab.get(off.cod_curso) if off.is_lab else None
        extra_fixed = _find_fixed_assignment(off, cfg.fixed_course_room, cfg.fixed_teacher_course)
        if extra_fixed:
            fixed = extra_fixed

        domains[idx] = OfferDomain(
            teacher_ids=valid_teachers,
            room_ids=room_ids,
            allowed_day_idxs=allowed_day_idxs,
            allowed_start_slots=allowed_start_slots,
            fixed_assignment=fixed,
            turn=off.turno,
        )

    return domains
