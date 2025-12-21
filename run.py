# run_demo.py
from typing import Dict
import pandas as pd

from src.config import GAConfig, TURN_SLOTS
from src.data_loader import load_data
from src.model import CourseOffering
from src.domains import build_offer_domains
from src.initial_population import build_initial_population
from src.evaluation import evaluate

def build_offers(bundle) -> list[CourseOffering]:
    # Join clase_fija con cursos para traer nombre/ciclo/turno_sugerido
    cf = bundle.clase_fija.merge(bundle.cursos, on="cod_curso", how="left")

    offers: list[CourseOffering] = []
    for _, r in cf.iterrows():
        tipo = str(r["tipo_hora"]).strip().upper()
        is_lab = (tipo == "L")

        # weekly_hours: AJUSTA esto a tu criterio real (por grupo horario/plan)
        # Por defecto: T=3, P=2, L=2 (cambia si tu doc/plan indica otra cosa)
        if tipo == "T":
            wh = 3
        elif tipo == "P":
            wh = 2
        else:
            wh = 2

        offers.append(CourseOffering(
            cod_curso=str(r["cod_curso"]),
            nombre=str(r.get("nombre", "")),
            ciclo=int(r.get("ciclo", 1)) if pd.notna(r.get("ciclo", 1)) else 1,
            turno=str(r.get("turno_sugerido", "M")),
            grupo_horario=str(r.get("grupo_horario", "01S")),
            tipo_hora=tipo,
            weekly_hours=wh,
            is_lab=is_lab
        ))
    return offers

def build_baseline_lab(bundle) -> Dict[str, dict]:
    """
    Para labs: si en programacion_2012B.csv el curso usa aula 'L..' lo fijamos.
    Nota: aquí el mapeo de 'hora_inicio' a 'slot' depende de tu grilla.
    Dejo un mapeo simple por ejemplo: si horas.csv tiene slot_id y rango_hora, puedes refinar.
    """
    baseline = {}
    prog = bundle.programacion.copy()

    # Normaliza día
    prog["dia_norm"] = prog["dia"].astype(str).str.upper().str.strip()

    # room_id desde cod_aula
    # si cod_aula es tipo "L6" o "A12", extraemos dígitos
    prog["room_id"] = prog["cod_aula"].astype(str).str.replace("A", "", regex=False).str.replace("L", "", regex=False)
    prog["room_id"] = pd.to_numeric(prog["room_id"], errors="coerce").fillna(0).astype(int)

    # teacher_id desde cod_docente
    prog["teacher_id"] = prog["cod_docente"].astype(str).str.replace("D", "", regex=False)
    prog["teacher_id"] = pd.to_numeric(prog["teacher_id"], errors="coerce").fillna(0).astype(int)

    # OJO: start/length aquí es un placeholder.
    # Si quieres exactitud: crea un mapeo hora_inicio->slot con horas.csv.
    def simple_time_to_slot(h: str) -> int:
        h = str(h).strip()
        # ejemplo súper simple: 08:00->0, 09:00->1...
        try:
            hh = int(h.split(":")[0])
            return max(0, min(14, hh - 8))
        except:
            return 0

    def simple_len(h_ini: str, h_fin: str) -> int:
        try:
            hi = int(str(h_ini).split(":")[0])
            hf = int(str(h_fin).split(":")[0])
            L = max(1, hf - hi)
            return min(5, L)
        except:
            return 2

    # filtrar labs por aula que empieza con "L"
    labs = prog[prog["cod_aula"].astype(str).str.upper().str.startswith("L")]
    for cod_curso, sub in labs.groupby("cod_curso"):
        # toma la primera fila como baseline fija
        row = sub.iloc[0]
        baseline[str(cod_curso)] = {
            "day_idx": 1,  # AJUSTA con mapping real de día (Martes=1 etc.)
            "start": simple_time_to_slot(row["hora_inicio"]),
            "length": simple_len(row["hora_inicio"], row["hora_termino"]),
            "room_id": int(row["room_id"]),
            "teacher_id": int(row["teacher_id"])
        }
    return baseline

def main():
    cfg = GAConfig()

    # 1) Cargar data (descomprime los CSV en ./data)
    bundle = load_data("data")

    # 2) Construir offerings (clase fija + cursos)
    offers = build_offers(bundle)

    # 3) Baseline labs (si existe)
    baseline_lab = build_baseline_lab(bundle)

    # 4) Dominios
    domains = build_offer_domains(
        offers=offers,
        aulas_df=bundle.aulas,
        pref_df=bundle.pref_docente_curso,
        turn_slots=TURN_SLOTS,
        n_days=cfg.N_DAYS,
        baseline_lab=baseline_lab
    )

    # 5) Población inicial (doc menciona 102, pero prueba con 20)
    pop = build_initial_population(
        offers=offers,
        domains=domains,
        n_slots_per_day=cfg.N_SLOTS_PER_DAY,
        pop_size=20
    )

    # 6) Evaluar
    # tamaños máximos (ajusta a tu data real)
    n_cycles = max(int(o.ciclo) for o in offers) if offers else 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    n_teachers = int(bundle.docentes["teacher_id"].max())

    for ind in pop:
        evaluate(ind, offers, n_cycles=n_cycles, n_rooms=n_rooms, n_teachers=n_teachers, cfg=cfg)

    best = max(pop, key=lambda x: x.fitness)
    print("Mejor penalización:", best.penalty, "fitness:", best.fitness)

    # Muestra primeras 5 asignaciones
    for off, g in list(zip(offers, best.genes))[:5]:
        print(off.cod_curso, off.tipo_hora, "-> doc", g.teacher_id, "day(s)", g.days, "room1", g.room1, "start", g.start1, "len", g.len1, "frozen", g.frozen)

if __name__ == "__main__":
    main()
