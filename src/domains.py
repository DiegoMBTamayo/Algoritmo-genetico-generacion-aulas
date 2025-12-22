# src/domains.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
from .model import CourseOffering

@dataclass(frozen=True)
class OfferDomain:
    teacher_ids: List[int]
    room_ids: List[int]
    allowed_day_idxs: List[int]
    allowed_start_slots: List[int]
    fixed_assignment: Optional[dict] = None

def split_weekly_hours(hours: int) -> Tuple[int, int]:
    if hours <= 1:
        return (hours, 0)
    a = (hours + 1) // 2
    b = hours // 2
    return (a, b)

def get_allowed_turns(cod_dispo: int) -> Set[str]:
    """
    Mapea el código de disponibilidad (docentes.csv) a los turnos permitidos (M, T, N).
    Según Tabla 3.4 del documento:
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
    return mapping.get(cod_dispo, set())

def build_offer_domains(
    offers: List[CourseOffering],
    aulas_df: pd.DataFrame,
    docentes_df: pd.DataFrame,  # <--- Nuevo parámetro: Tabla de docentes completa
    turn_slots: Dict[str, List[int]],
    n_days: int,
    baseline_lab: Dict[str, dict]
) -> Dict[int, OfferDomain]:

    domains: Dict[int, OfferDomain] = {}

    # 1. Clasificar aulas (Labs vs Teoría)
    lab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == True, "room_id"].dropna().unique().tolist()
    nonlab_rooms = aulas_df.loc[aulas_df["es_laboratorio"] == False, "room_id"].dropna().unique().tolist()

    # 2. Pre-procesar disponibilidad docente
    # Crear un diccionario: teacher_id -> Set de turnos permitidos {'M', 'T', ...}
    teacher_availability = {}
    for _, row in docentes_df.iterrows():
        tid = int(str(row["cod_docente"]).replace("D", ""))
        cdispo = int(row["cod_dispo"]) if pd.notna(row["cod_dispo"]) else 6
        teacher_availability[tid] = get_allowed_turns(cdispo)
        
    all_teachers = list(teacher_availability.keys())

    for idx, off in enumerate(offers):
        # --- A. Filtrado de Docentes por Disponibilidad ---
        # El docente debe tener disponible el turno del curso (off.turno)
        valid_teachers = []
        for tid in all_teachers:
            if off.turno in teacher_availability[tid]:
                valid_teachers.append(tid)
        
        # Fallback: Si nadie puede (error de datos), asignamos todos para evitar crash
        if not valid_teachers:
            valid_teachers = all_teachers

        # --- B. Filtrado de Aulas (Lab vs Teoría) ---
        # Documento ROE2 y ROE3: Teoría -> Aula normal, Lab -> Aula lab
        room_ids = lab_rooms if off.is_lab else nonlab_rooms

        # --- C. Slots permitidos según turno ---
        allowed_day_idxs = list(range(n_days))
        allowed_start_slots = turn_slots.get(off.turno, [])

        # --- D. Manejo de Laboratorios Fijos (Baseline) ---
        fixed = None
        if off.is_lab and off.cod_curso in baseline_lab:
            fixed = baseline_lab[off.cod_curso].copy()

        domains[idx] = OfferDomain(
            teacher_ids=valid_teachers,
            room_ids=room_ids,
            allowed_day_idxs=allowed_day_idxs,
            allowed_start_slots=allowed_start_slots,
            fixed_assignment=fixed
        )

    return domains