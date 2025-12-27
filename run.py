import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from src.config import GAConfig, load_config, resolve_turn_for_cycle
from src.data_loader import load_data
from src.model import CourseOffering
from src.domains import build_offer_domains
from src.initial_population import build_initial_population
from src.ga import GeneticSolver
from src.evaluation import evaluate, EvaluationResult
from src.operators import repair_individual
from src.encoding import gene_to_bits


# Compat helpers usados en app.py
def to_binary_string(val, bits):
    if val is None:
        return "0" * bits
    return format(int(val), f"0{bits}b")


def day_to_bitmask(day_idxs):
    mask = ["0"] * 6
    for d in day_idxs:
        if 0 <= d < 6:
            mask[d] = "1"
    return "".join(mask)


def build_offers(bundle, cfg: GAConfig) -> List[CourseOffering]:
    cf = bundle.clase_fija.merge(bundle.cursos, left_on="cod_curso", right_on="ID_Curso", how="left")
    offers: List[CourseOffering] = []
    for _, r in cf.iterrows():
        tipo = str(r["tipo_hora"]).strip().upper()
        is_lab = tipo == "L"
        wh = cfg.weekly_hours_by_type.get(tipo, 2)

        ciclo_val = int(r.get("Ciclo", r.get("ciclo", 1)))
        turno_calc = resolve_turn_for_cycle(ciclo_val, cfg.cycle_turn_map)

        offers.append(
            CourseOffering(
                cod_curso=str(r["cod_curso"]),
                nombre=str(r.get("Nombre", "Curso")),
                ciclo=ciclo_val,
                turno=turno_calc,
                grupo_horario=str(r.get("grupo_horario", "01S")),
                tipo_hora=tipo,
                weekly_hours=wh,
                is_lab=is_lab,
            )
        )
    return offers


def build_baseline_lab(bundle):
    return {}


def _slot_range(slot_idx: int, length: int, horas_df: pd.DataFrame) -> Tuple[str, str]:
    start_row = horas_df.loc[horas_df["slot_id"] == slot_idx]
    end_row = horas_df.loc[horas_df["slot_id"] == slot_idx + length - 1]
    start = start_row.iloc[0]["rango_hora"].split("-")[0].strip() if not start_row.empty else str(slot_idx)
    end = end_row.iloc[0]["rango_hora"].split("-")[-1].strip() if not end_row.empty else str(slot_idx + length)
    return start, end


def individual_to_dataframe(best, offers, bundle, cfg: GAConfig) -> pd.DataFrame:
    dias_map = {int(r.day_idx): r.dia for r in bundle.dias.itertuples()}
    data = []
    for g, off in zip(best.genes, offers):
        blocks = [
            (g.days[0] if len(g.days) > 0 else None, g.start1, g.len1, g.room1, "1"),
            (g.days[1] if len(g.days) > 1 else None, g.start2, g.len2, g.room2, "2"),
        ]
        for day, start, length, room, sesion in blocks:
            if day is None or start is None or length in (None, 0):
                continue
            h_ini, h_fin = _slot_range(start, length, bundle.horas)
            room_code = bundle.aulas.loc[bundle.aulas["room_id"] == room, "cod_aula"].iloc[0]
            doc_row = bundle.docentes.loc[bundle.docentes["teacher_id"] == g.teacher_id]
            docente = doc_row.iloc[0]["nombres"] if not doc_row.empty else f"Doc {g.teacher_id}"
            data.append(
                {
                    "Ciclo": off.ciclo,
                    "Curso": off.cod_curso,
                    "Nombre": off.nombre,
                    "Grupo": off.grupo_horario,
                    "Tipo": off.tipo_hora,
                    "Docente": docente,
                    "Aula": room_code,
                    "Dia": dias_map.get(day, str(day)),
                    "Hora_Inicio": h_ini,
                    "Hora_Fin": h_fin,
                    "Sesion": sesion,
                }
            )
    return pd.DataFrame(data)


def print_chromosome_representation(individual, offers):
    print("\n" + "=" * 80)
    print("REPRESENTACIÓN DE CROMOSOMA - ETIQUETA + BITS (30 bits)")
    print("ETIQUETA: Curso Grupo Tipo | BITS: Doc(6) Días(6) Aula1(4) Hora1(5) Aula2(4) Hora2(5)")
    print("=" * 80)
    for i, (gene, off) in enumerate(zip(individual.genes, offers)):
        if i >= 20:
            break
        etiqueta = f"{off.cod_curso} {off.grupo_horario} {off.tipo_hora}"
        bits = gene_to_bits(gene).as_string()
        print(f"{etiqueta:<18} {bits}")
    print("=" * 80 + "\n")


def export_outputs(df_schedule: pd.DataFrame, eval_res: EvaluationResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_schedule.to_csv(out_dir / "schedule.csv", index=False)
    conflicts = pd.DataFrame(
        [
            {"tipo": "hi_ciclo", "valor": eval_res.hi},
            {"tipo": "ai_aula", "valor": eval_res.ai},
            {"tipo": "pi_docente", "valor": eval_res.pi},
            {"tipo": "penalizacion", "valor": eval_res.penalty},
        ]
    )
    conflicts.to_csv(out_dir / "conflicts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Ejecución end-to-end del AG de horarios")
    parser.add_argument("--config", default="config.yaml", help="Ruta al archivo de configuración")
    parser.add_argument("--data_dir", default="data", help="Directorio con los CSV de entrada")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Cargando datos...")
    bundle = load_data(args.data_dir)
    if "es_laboratorio" not in bundle.aulas.columns:
        bundle.aulas["es_laboratorio"] = bundle.aulas["cod_tipo"].astype(str).str.startswith("L")

    offers = build_offers(bundle, cfg)
    baseline_lab = build_baseline_lab(bundle)

    domains = build_offer_domains(
        offers=offers,
        aulas_df=bundle.aulas,
        docentes_df=bundle.docentes,
        cfg=cfg,
        baseline_lab=baseline_lab,
    )

    pop = build_initial_population(offers, domains, cfg.n_slots_per_day, cfg.population_size)
    pop = [repair_individual(ind, offers, domains, cfg) for ind in pop]

    n_cycles = 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    if "teacher_id" not in bundle.docentes.columns:
        bundle.docentes["teacher_id"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    n_teachers = int(bundle.docentes["teacher_id"].max()) + 1

    solver = GeneticSolver(offers, domains, cfg, n_cycles, n_rooms, n_teachers)

    print(f"Generaciones: {cfg.generations} | Población: {cfg.population_size}")
    start = time.perf_counter()
    best_ind = solver.evolve(pop, cfg.generations, cfg.mutation_rate)
    elapsed = time.perf_counter() - start

    eval_res = evaluate(best_ind, offers, n_cycles, n_rooms, n_teachers, cfg)

    print("\n--- MEJOR SOLUCIÓN ---")
    print(f"Fitness: {best_ind.fitness:.5f} | Penalización: {best_ind.penalty} | Tiempo: {elapsed:.2f}s")
    print(f"hi={eval_res.hi} ai={eval_res.ai} pi={eval_res.pi}")
    print_chromosome_representation(best_ind, offers)

    df_schedule = individual_to_dataframe(best_ind, offers, bundle, cfg)
    out_dir = Path("outputs")
    export_outputs(df_schedule, eval_res, out_dir)
    if solver.history:
        pd.DataFrame(solver.history).to_csv(out_dir / "history.csv", index=False)
    metrics = {
        "best_penalty": eval_res.penalty,
        "hi": eval_res.hi,
        "ai": eval_res.ai,
        "pi": eval_res.pi,
        "time_sec": elapsed,
        "generations_ran": len(solver.history),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    print("Se guardaron resultados en outputs/schedule.csv y outputs/conflicts.csv")


if __name__ == "__main__":
    main()
