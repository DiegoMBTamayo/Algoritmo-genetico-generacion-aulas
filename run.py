# run.py
import pandas as pd
from src.config import GAConfig, TURN_SLOTS
from src.data_loader import load_data
from src.model import CourseOffering
from src.domains import build_offer_domains
from src.initial_population import build_initial_population
from src.ga import GeneticSolver 

def build_offers(bundle) -> list[CourseOffering]:
    cf = bundle.clase_fija.merge(bundle.cursos, left_on="cod_curso", right_on="ID_Curso", how="left")
    offers: list[CourseOffering] = []
    for _, r in cf.iterrows():
        tipo = str(r["tipo_hora"]).strip().upper()
        is_lab = (tipo == "L")
        if tipo == "T": wh = 3
        elif tipo == "P": wh = 2
        else: wh = 2

        try:
            ciclo_val = int(r.get("Ciclo", 1))
        except:
            ciclo_val = 1
        
        if ciclo_val <= 3: turno_calc = "M"
        elif ciclo_val <= 7: turno_calc = "T"
        else: turno_calc = "N"

        offers.append(CourseOffering(
            cod_curso=str(r["cod_curso"]),
            nombre=str(r.get("Nombre", "Curso")),
            ciclo=ciclo_val,
            turno=turno_calc,
            grupo_horario=str(r.get("grupo_horario", "01S")), # Asegura que esto exista en tu CSV clase_fija
            tipo_hora=tipo,
            weekly_hours=wh,
            is_lab=is_lab
        ))
    return offers

def build_baseline_lab(bundle):
    return {} 

# --- VISUALIZACIÓN CORREGIDA (ETIQUETA + 30 BITS) ---

def to_binary_string(val, bits):
    if val is None: return "0" * bits
    return format(val, f'0{bits}b')

def day_to_bitmask(day_idxs):
    mask = ["0"] * 6
    for d in day_idxs:
        if 0 <= d < 6: mask[d] = "1"
    return "".join(mask)

def print_chromosome_representation(individual, offers):
    print("\n" + "="*80)
    print("REPRESENTACIÓN DE CROMOSOMA - FORMATO: ETIQUETA + BITS (30 bits)")
    print("ETIQUETA: Curso Grupo Tipo")
    print("BITS:     Docente(6) Días(6) Aula1(4) Hora1(5) Aula2(4) Hora2(5)")
    print("="*80)

    for i, (gene, off) in enumerate(zip(individual.genes, offers)):
        if i >= 20: break 
        
        # 1. Construir la ETIQUETA (Parte Fija)
        # Ejemplo: C01 01S T
        etiqueta = f"{off.cod_curso} {off.grupo_horario} {off.tipo_hora}"
        
        # 2. Construir el CROMOSOMA (Parte Variable - 30 bits)
        doc_bin = to_binary_string(gene.teacher_id, 6)
        dias_bin = day_to_bitmask(gene.days)
        aula1_bin = to_binary_string(gene.room1, 4)
        hora1_bin = to_binary_string(gene.start1, 5)
        aula2_bin = to_binary_string(gene.room2, 4)
        hora2_bin = to_binary_string(gene.start2, 5)

        # Concatenamos todo junto
        cromosoma = f"{doc_bin}{dias_bin}{aula1_bin}{hora1_bin}{aula2_bin}{hora2_bin}"

        # Imprimir alineado
        print(f"{etiqueta:<12} {cromosoma}")

    print("="*80 + "\n")

# --- MAIN ---

def main():
    cfg = GAConfig()
    print("Cargando datos...")
    bundle = load_data("data")
    
    if "es_laboratorio" not in bundle.aulas.columns:
        bundle.aulas["es_laboratorio"] = bundle.aulas["cod_tipo"].astype(str).str.startswith("L")

    offers = build_offers(bundle)
    print(f"Se generaron {len(offers)} ofertas (genes).")
    
    baseline_lab = build_baseline_lab(bundle)

    print("Construyendo dominios...")
    domains = build_offer_domains(
        offers=offers,
        aulas_df=bundle.aulas,
        docentes_df=bundle.docentes, 
        turn_slots=TURN_SLOTS,
        n_days=cfg.N_DAYS,
        baseline_lab=baseline_lab
    )

    POP_SIZE = 50       
    GENERATIONS = 100   
    MUTATION_RATE = 0.1 

    print("Generando población inicial...")
    pop = build_initial_population(offers, domains, cfg.N_SLOTS_PER_DAY, POP_SIZE)

    print(f"Iniciando evolución ({GENERATIONS} gens)...")
    
    n_cycles = 10 
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    if "tid_num" not in bundle.docentes.columns:
        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    n_teachers = int(bundle.docentes["tid_num"].max()) + 1

    solver = GeneticSolver(offers, domains, cfg, n_cycles, n_rooms, n_teachers)
    best_ind = solver.evolve(pop, GENERATIONS, MUTATION_RATE)

    print("\n--- MEJOR SOLUCIÓN ---")
    print(f"Fitness: {best_ind.fitness:.5f} | Penalización: {best_ind.penalty}")
    
    # Llamada a la visualización corregida
    print_chromosome_representation(best_ind, offers)

if __name__ == "__main__":
    main()