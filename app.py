# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import copy
import random
from src.data_loader import load_data
from src.config import GAConfig, TURN_SLOTS
from src.domains import build_offer_domains
from src.initial_population import build_initial_population, random_gene_for_offer
from src.ga import GeneticSolver
from src.evaluation import evaluate
from run import build_offers, build_baseline_lab, to_binary_string, day_to_bitmask

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema de Horarios FIIS", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS MEJORADOS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 5px;
        height: 50px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .styled-table {
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 0.8em;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-width: 400px;
        width: 100%;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        border-radius: 5px 5px 0 0;
        overflow: hidden;
    }
    .styled-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
        font-weight: bold;
    }
    .styled-table th, .styled-table td {
        padding: 6px 8px;
        border: 1px solid #ddd;
    }
    .styled-table tbody tr {
        background-color: #ffffff;
        color: #333;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .scroll-table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        background-color: #fff;
        margin-bottom: 20px;
        border-radius: 5px;
    }
    .scroll-table-container thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: #009879;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #333;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-highlight {
        color: #d32f2f;
    }
    .penalty-list {
        height: 150px;
        overflow-y: auto;
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 10px;
        font-size: 11px;
        font-family: monospace;
        color: #333;
        border-radius: 4px;
    }
    .subpop-header {
        background-color: #333;
        color: #fff;
        padding: 8px;
        font-weight: bold;
        font-size: 14px;
        margin-top: 15px;
        margin-bottom: 0px;
        border-radius: 5px 5px 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES HELPERS ---
def get_docente_name(bundle, teacher_id):
    row = bundle.docentes[bundle.docentes["teacher_id"] == teacher_id]
    if not row.empty:
        r = row.iloc[0]
        return f"{r['nombres']} {r['ap_paterno']}"
    return f"Docente {teacher_id}"

def get_docente_code(bundle, teacher_id):
    row = bundle.docentes[bundle.docentes["teacher_id"] == teacher_id]
    return row.iloc[0]['cod_docente'] if not row.empty else f"D{teacher_id}"

def get_room_code(bundle, room_id):
    row = bundle.aulas[bundle.aulas["room_id"] == room_id]
    return row.iloc[0]['cod_aula'] if not row.empty else f"Aula {room_id}"

def get_html_card(nombre, grupo, docente, aula, tipo):
    return (
        f"<div style='border:1px solid #ccc; margin-bottom:4px; font-family:sans-serif; overflow:hidden; border-radius:3px;'>"
        f"<div style='background-color:#e0f7fa; color:#006064; padding:2px 4px; font-size:10px; font-weight:bold; border-bottom:1px solid #b2ebf2;'>{nombre}</div>"
        f"<div style='background-color:#fff; color:#333; padding:2px 4px; font-size:9px;'><b>{grupo}</b> {docente}<br>{aula} ({tipo})</div>"
        f"</div>"
    )

def create_schedule_matrix(best_ind, offers, bundle, filter_mode, filter_val, dias_map, horas_map):
    TOTAL_SLOTS = 17 
    matrix = np.full((TOTAL_SLOTS, 6), "", dtype=object)
    
    for g, off in zip(best_ind.genes, offers):
        show = False
        if filter_mode == "Ciclo" and str(off.ciclo) == str(filter_val): show = True
        if filter_mode == "Aula" and (g.room1 == filter_val or g.room2 == filter_val): show = True
        
        if not show: continue

        doc_code = get_docente_code(bundle, g.teacher_id)
        room1_code = get_room_code(bundle, g.room1)
        
        if filter_mode == "Ciclo" or g.room1 == filter_val:
            for d in [g.days[0]]:
                if 0 <= d < 6:
                    for h in range(g.start1, g.start1 + g.len1):
                        if 0 <= h < TOTAL_SLOTS:
                            matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room1_code, off.tipo_hora)
        
        if len(g.days) > 1 and g.room2 is not None:
            if filter_mode == "Ciclo" or g.room2 == filter_val:
                room2_code = get_room_code(bundle, g.room2)
                for d in [g.days[1]]:
                    if 0 <= d < 6:
                        for h in range(g.start2, g.start2 + g.len2):
                            if 0 <= h < TOTAL_SLOTS:
                                matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room2_code, off.tipo_hora)
    
    cols = [dias_map.get(i, f"Dia {i}") for i in range(6)]
    df_mat = pd.DataFrame(matrix, columns=cols)
    df_mat.index = [horas_map.get(i, f"Slot {i}") for i in range(TOTAL_SLOTS)]
    return df_mat

# --- HELPER: MATRICES DE CONFLICTO ---
def get_conflict_matrices(individual, offers, bundle):
    n_cycles = 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    n_teachers = int(bundle.docentes["teacher_id"].max()) + 1
    TOTAL_SLOTS = 17

    count_cycle = np.zeros((n_cycles + 1, 6, TOTAL_SLOTS), dtype=int)
    count_room = np.zeros((n_rooms + 1, 6, TOTAL_SLOTS), dtype=int)
    count_teach = np.zeros((n_teachers + 1, 6, TOTAL_SLOTS), dtype=int)

    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        cyc = int(off.ciclo) - 1
        for h in range(g.start1, g.start1 + g.len1):
            if h < TOTAL_SLOTS and g.days[0] < 6:
                count_cycle[cyc, g.days[0], h] += 1
                count_room[g.room1, g.days[0], h] += 1
                count_teach[g.teacher_id, g.days[0], h] += 1
        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < TOTAL_SLOTS and g.days[1] < 6:
                    count_cycle[cyc, g.days[1], h] += 1
                    count_room[g.room2, g.days[1], h] += 1
                    count_teach[g.teacher_id, g.days[1], h] += 1
    
    return count_cycle, count_room, count_teach

# --- HELPER: CÁLCULO DINÁMICO DE PENALIDADES (GLOBAL) ---
def calculate_dynamic_details(individual, offers, bundle):
    count_cycle, count_room, count_teach = get_conflict_matrices(individual, offers, bundle)
    cfg = GAConfig()
    
    gene_stats = []
    penalties_docente = []
    penalties_doc_ciclo = []
    penalties_doc_aula = []
    total_fitness_sum = 0.0
    
    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        pen_gen = 0
        pen_teacher = 0
        pen_cycle = 0
        pen_room = 0
        cyc = int(off.ciclo) - 1
        
        # Bloque 1
        for h in range(g.start1, g.start1 + g.len1):
            if h < 17 and g.days[0] < 6:
                if count_cycle[cyc, g.days[0], h] > 1: pen_cycle += cfg.W_CONFLICT_CYCLE
                if count_room[g.room1, g.days[0], h] > 1: pen_room += cfg.W_CONFLICT_ROOM
                if count_teach[g.teacher_id, g.days[0], h] > 1: pen_teacher += cfg.W_CONFLICT_TEACHER

        # Bloque 2
        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < 17 and g.days[1] < 6:
                    if count_cycle[cyc, g.days[1], h] > 1: pen_cycle += cfg.W_CONFLICT_CYCLE
                    if count_room[g.room2, g.days[1], h] > 1: pen_room += cfg.W_CONFLICT_ROOM
                    if count_teach[g.teacher_id, g.days[1], h] > 1: pen_teacher += cfg.W_CONFLICT_TEACHER
        
        pen_gen = pen_cycle + pen_room + pen_teacher
        aptitud = 1.0 / (1.0 + float(pen_gen))
        
        total_fitness_sum += aptitud
        
        gene_stats.append({"idx": i, "penal": pen_gen, "aptitud": aptitud})
        
        doc_code = get_docente_code(bundle, g.teacher_id)
        if pen_teacher > 0: penalties_docente.append(f"{off.cod_curso} {doc_code}: {pen_teacher}")
        if pen_cycle > 0: penalties_doc_ciclo.append(f"{off.cod_curso} (Ciclo {off.ciclo}): {pen_cycle}")
        if pen_room > 0: penalties_doc_aula.append(f"{off.cod_curso} {doc_code}: {pen_room}")

    return gene_stats, total_fitness_sum, penalties_docente, penalties_doc_ciclo, penalties_doc_aula

# --- LOGICA DE EVOLUCIÓN OPTIMIZADA (TORNEO) ---
def tournament_selection(population, k=4):
    candidates = random.sample(population, k)
    return max(candidates, key=lambda x: x.fitness)

def smart_mutate(individual, offers, domains, mutation_rate, cfg):
    if random.random() > mutation_rate:
        return

    idx = random.randint(0, len(individual.genes) - 1)
    if individual.genes[idx].frozen: return

    # Mutación simple
    new_gene = random_gene_for_offer(offers[idx], domains[idx], cfg.N_SLOTS_PER_DAY)
    individual.genes[idx] = new_gene

def run_smart_generation(solver, population, mutation_rate, offers, cfg):
    pop_size = len(population)
    elite_count = int(pop_size * 0.10)
    if elite_count < 1: elite_count = 1
    
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_pop = [copy.deepcopy(ind) for ind in population[:elite_count]]
    
    while len(new_pop) < pop_size:
        p1 = tournament_selection(population, k=4)
        p2 = tournament_selection(population, k=4)
        
        child = solver.crossover_subpopulations(p1, p2)
        smart_mutate(child, offers, solver.domains, mutation_rate, cfg)
        evaluate(child, offers, solver.n_cycles, solver.n_rooms, solver.n_teachers, cfg)
        new_pop.append(child)
    
    return new_pop

# --- MAIN APP ---
def main():
    if 'bundle' not in st.session_state:
        st.session_state.bundle = load_data("data")
    bundle = st.session_state.bundle

    dias_df = bundle.dias
    dias_map = dict(zip(dias_df['day_idx'], dias_df['dia']))
    horas_df = bundle.horas
    horas_map = dict(zip(horas_df['slot_id'], horas_df['rango_hora']))

    if 'generation_count' not in st.session_state: st.session_state.generation_count = 0
    if 'history' not in st.session_state: st.session_state.history = []
    if 'population' not in st.session_state: st.session_state.population = None
    if 'best_ind' not in st.session_state: st.session_state.best_ind = None
    if 'initial_pop_list' not in st.session_state: st.session_state.initial_pop_list = None 

    APP_CONFIG = GAConfig(N_SLOTS_PER_DAY=17) 

    with st.sidebar:
        st.title("🧬 Menú Principal")
        st.markdown("---")
        page = st.radio("Ir a la sección:", [
            "Realizar Asignación", 
            "Procesos Adicionales", 
            "Matrices de Conflictos",
            "Editor Manual",
            "Horario por Ciclo", 
            "Horario por Aula", 
            "Horarios en General"
        ])
        st.markdown("---")
        st.info("Sistema de Optimización de Horarios\nAlgoritmo Genético")

    # 1. REALIZAR ASIGNACIÓN
    if page == "Realizar Asignación":
        st.header("📋 Realizar Asignación y Gestión de Datos")
        if st.session_state.get('show_save_success', False):
            st.success("¡Datos actualizados y recargados correctamente!")
            st.session_state['show_save_success'] = False

        tab_listas = st.tabs(["Docentes", "Aulas", "Cursos", "Clase Fija"])
        
        with tab_listas[0]:
            cols_csv_docentes = ["cod_docente", "cod_dispo", "ap_paterno", "ap_materno", "nombres"]
            edited_docentes = st.data_editor(bundle.docentes[cols_csv_docentes], num_rows="dynamic", key="editor_docentes", height=250, use_container_width=True)
        with tab_listas[1]:
            cols_csv_aulas = ["cod_aula", "cod_tipo", "capacidad"]
            edited_aulas = st.data_editor(bundle.aulas[cols_csv_aulas], num_rows="dynamic", key="editor_aulas", height=250, use_container_width=True)
        with tab_listas[2]:
            cols_csv_cursos = ["ID_Curso", "Nombre", "Ciclo"]
            edited_cursos = st.data_editor(bundle.cursos[cols_csv_cursos], num_rows="dynamic", key="editor_cursos", height=250, use_container_width=True)
        with tab_listas[3]:
            cols_csv_fija = ["cod_curso", "grupo_horario", "tipo_hora"]
            edited_fija = st.data_editor(bundle.clase_fija[cols_csv_fija], num_rows="dynamic", key="editor_fija", height=250, use_container_width=True)

        if st.button("💾 Guardar Cambios en Archivos CSV"):
            try:
                edited_docentes.to_csv("data/docentes.csv", index=False)
                edited_aulas.to_csv("data/aulas.csv", index=False)
                edited_cursos.to_csv("data/cursos.csv", index=False)
                edited_fija.to_csv("data/clase_fija.csv", index=False)
                st.session_state.bundle = load_data("data")
                st.session_state['show_save_success'] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar archivos: {e}")

        st.divider()
        if st.button("🚀 CREAR POBLACIÓN INICIAL (PROBABILÍSTICA)"):
            st.session_state.population = None
            st.session_state.generation_count = 0
            st.session_state.history = []
            
            with st.status("Generando ofertas y población inicial...", expanded=True) as status:
                offers = build_offers(bundle)
                baseline = build_baseline_lab(bundle)
                aulas_calc = bundle.aulas.copy()
                aulas_calc["es_laboratorio"] = aulas_calc["cod_tipo"].astype(str).str.startswith("L")
                
                domains = build_offer_domains(offers, aulas_calc, bundle.docentes, TURN_SLOTS, 6, baseline)
                
                # INICIALIZACIÓN PROBABILÍSTICA (ALEATORIA)
                initial_pop = build_initial_population(offers, domains, APP_CONFIG.N_SLOTS_PER_DAY, 102)
                
                n_cycles = 10
                n_rooms = int(bundle.aulas["room_id"].max()) + 1
                n_teachers = int(bundle.docentes["teacher_id"].max()) + 1
                cfg = GAConfig()
                for ind in initial_pop:
                    evaluate(ind, offers, n_cycles, n_rooms, n_teachers, APP_CONFIG)
                
                st.session_state.initial_pop_list = initial_pop
                st.session_state.population = initial_pop
                st.session_state.offers = offers
                st.session_state.domains = domains
                st.session_state.solver = GeneticSolver(offers, domains, APP_CONFIG, n_cycles, n_rooms, n_teachers)
                
                best = max(initial_pop, key=lambda x: x.fitness)
                st.session_state.best_ind = best
                st.session_state.history.append((0, int(best.penalty)))
                st.session_state.has_run = True
                
                status.update(label="¡Población Generada!", state="complete", expanded=False)

        if st.session_state.get("has_run", False) and st.session_state.initial_pop_list:
            st.divider()
            st.subheader("Muestra de Población Inicial (Aleatoria)")
            sample_ind = st.session_state.initial_pop_list[0]
            offers = st.session_state.offers
            pop_data = []
            for g, off in zip(sample_ind.genes, offers):
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                bits_a1 = to_binary_string(g.room1, 4)
                bits_h1 = to_binary_string(g.start1, 5)
                bits_a2 = to_binary_string(g.room2, 4) if (len(g.days)>1 and g.room2 is not None) else "0000"
                bits_h2 = to_binary_string(g.start2, 5) if (len(g.days)>1 and g.room2 is not None) else "00000"
                pop_data.append({
                    "Curso": off.cod_curso, "Grupo": off.grupo_horario,
                    "Doc(6b)": bits_doc, "Días(6b)": bits_dias,
                    "Aula1(4b)": bits_a1, "Hora1(5b)": bits_h1,
                    "Aula2(4b)": bits_a2, "Hora2(5b)": bits_h2
                })
            st.dataframe(pd.DataFrame(pop_data), height=300, use_container_width=True)

    # 2. PROCESOS ADICIONALES
    elif page == "Procesos Adicionales":
        st.header("⚙️ Procesos Adicionales")
        
        if not st.session_state.get("has_run", False):
            st.warning("⚠️ Primero ejecute 'CREAR POBLACIÓN INICIAL' en la pestaña 'Realizar Asignación'.")
        else:
            solver = st.session_state.solver
            offers = st.session_state.offers
            cfg = APP_CONFIG

            c_ctrl, c_metrics = st.columns([2, 1])
            with c_ctrl:
                c1, c2 = st.columns(2)
                if c1.button("🔄 Altera Poblac. Inicial (Reset)"):
                    with st.spinner("Reiniciando población (Aleatoria)..."):
                        pop_size = 102
                        # INICIALIZACIÓN PROBABILÍSTICA (ALEATORIA)
                        new_pop = build_initial_population(offers, solver.domains, cfg.N_SLOTS_PER_DAY, pop_size)
                        for ind in new_pop:
                            evaluate(ind, offers, solver.n_cycles, solver.n_rooms, solver.n_teachers, cfg)
                        st.session_state.initial_pop_list = new_pop
                        st.session_state.population = new_pop
                        st.session_state.generation_count = 0
                        st.session_state.history = []
                        best = max(new_pop, key=lambda x: x.fitness)
                        st.session_state.best_ind = best
                        st.session_state.history.append((0, int(best.penalty)))
                        st.rerun()

                if c2.button("📊 Calcular Penalidad Actual"):
                    for ind in st.session_state.population:
                        evaluate(ind, offers, solver.n_cycles, solver.n_rooms, solver.n_teachers, cfg)
                    best = max(st.session_state.population, key=lambda x: x.fitness)
                    st.session_state.best_ind = best
                    st.rerun()

                c3, c4 = st.columns(2)
                if c3.button("▶ Gen. 1 Población"):
                    current_pop = st.session_state.population
                    next_gen = run_smart_generation(solver, current_pop, 0.2, offers, cfg)
                    st.session_state.population = next_gen
                    st.session_state.generation_count += 1
                    best = max(next_gen, key=lambda x: x.fitness)
                    st.session_state.best_ind = best
                    st.session_state.history.append((st.session_state.generation_count, int(best.penalty)))
                    st.rerun()

                if c4.button("⏩ Gen. 50 Poblaciones"):
                    current_pop = st.session_state.population
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i in range(50):
                        current_pop = run_smart_generation(solver, current_pop, 0.2, offers, cfg)
                        st.session_state.generation_count += 1
                        best = max(current_pop, key=lambda x: x.fitness)
                        progress_bar.progress((i + 1) / 50)
                        status_text.text(f"Generando {i+1}/50... Penalidad: {int(best.penalty)}")
                    
                    st.session_state.population = current_pop
                    st.session_state.best_ind = best
                    st.session_state.history.append((st.session_state.generation_count, int(best.penalty)))
                    status_text.empty()
                    progress_bar.empty()
                    st.rerun()

            with c_metrics:
                gen_num = st.session_state.get('generation_count', 0)
                best_pen = int(st.session_state.best_ind.penalty) if st.session_state.best_ind else 0
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">POBLACIÓN ACTUAL</div>
                    <div class="metric-value">{gen_num}</div>
                    <hr style="margin: 10px 0; border-color: #eee;">
                    <div class="metric-label">PENALIDAD TOTAL</div>
                    <div class="metric-value metric-highlight">{best_pen}</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")
            best = st.session_state.best_ind
            # Calculamos detalles globales primero
            gene_stats, total_fit_sum, pen_doc, pen_ciclo, pen_aula = calculate_dynamic_details(best, offers, bundle)

            # 1. DETALLE POR CURSO
            st.subheader("1. Detalle por curso")
            
            # Encabezados actualizados
            html_main = """<table class="styled-table"><thead><tr>
            <th>Curso</th><th>Grupo</th><th>Docente</th><th>Cromosoma</th><th>Penal</th><th>Aptitud</th><th>D.G.</th><th>P.Selec</th><th>P.Acumul</th>
            </tr></thead><tbody>"""
            
            p_acum = 0.0
            
            for stat in gene_stats:
                idx = stat['idx']
                g = best.genes[idx]
                off = offers[idx]
                doc_code = get_docente_code(bundle, g.teacher_id)
                
                # --- CONSTRUCCIÓN DEL CROMOSOMA (Todos los bits separados) ---
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                
                # Bloque 1
                bits_aula1 = to_binary_string(g.room1, 4)
                bits_hora1 = to_binary_string(g.start1, 5)
                
                # Bloque 2 (si existe)
                if len(g.days) > 1 and g.room2 is not None:
                    bits_aula2 = to_binary_string(g.room2, 4)
                    bits_hora2 = to_binary_string(g.start2, 5)
                else:
                    bits_aula2 = "0000"
                    bits_hora2 = "00000"
                
                # Unimos todo en una sola cadena separada por espacios
                full_bits = f"{bits_doc} {bits_dias} {bits_aula1} {bits_hora1} {bits_aula2} {bits_hora2}"
                
                # Cálculos de probabilidad
                p_selec = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                p_acum += p_selec
                
                html_main += f"""<tr>
                <td>{off.cod_curso}</td><td>{off.grupo_horario}</td><td>{doc_code}</td>
                <td style='font-family:monospace; font-size:0.9em'>{full_bits}</td>
                <td>{int(stat['penal'])}</td>
                <td>{stat['aptitud']:.6f}</td>
                <td>{total_fit_sum:.2f}</td>
                <td>{p_selec:.6f}</td>
                <td>{p_acum:.6f}</td>
                </tr>"""
            html_main += "</tbody></table>"
            st.markdown(f"<div class='scroll-table-container'>{html_main}</div>", unsafe_allow_html=True)

           # 2. SUBPOBLACIONES
            st.subheader("2. Subpoblaciones")
            
            def render_subpop_table(turno_key, title):
                if turno_key not in solver.subpopulations: return ""
                
                full_html = f"<div class='subpop-header'>{title}</div>"
                
                for horas, idxs in solver.subpopulations[turno_key].items():
                    # 1. Calcular D.G. Local (Suma de aptitudes de ESTE grupo)
                    local_fit_sum = 0.0
                    local_rows_data = []
                    
                    for idx in idxs:
                        stat = next((s for s in gene_stats if s['idx'] == idx), None)
                        if stat: 
                            local_fit_sum += stat['aptitud']
                            local_rows_data.append((idx, stat))
                    
                    if not local_rows_data: continue

                    # 2. Construir tabla
                    # NOTA: Ahora la columna dice 'D.G. (Local)' y usa 'local_fit_sum'
                    html = f"""<div style='font-size:12px; font-weight:bold; margin:5px 0; color:#555;'>► Grupo de {horas} Horas</div>
                    <div class="scroll-table-container" style="height:auto; max_height:200px; margin-bottom:15px;">
                    <table class="styled-table" style="margin-top:0;">
                    <thead><tr>
                        <th>Carga</th><th>Curso</th><th>Grupo</th><th>Docente</th>
                        <th>Cromosoma</th>
                        <th>Penal</th><th>Aptitud</th><th>D.G. (Local)</th> <th>P.Sel (Local)</th><th>P.Acum (Local)</th>
                    </tr></thead><tbody>"""
                    
                    p_acum_local = 0.0
                    for idx, stat in local_rows_data:
                        g = best.genes[idx]
                        off = offers[idx]
                        doc_code = get_docente_code(bundle, g.teacher_id)
                        
                        # Construcción del Cromosoma
                        bits_doc = to_binary_string(g.teacher_id, 6)
                        bits_dias = day_to_bitmask(g.days)
                        bits_aula1 = to_binary_string(g.room1, 4)
                        bits_hora1 = to_binary_string(g.start1, 5)
                        
                        if len(g.days) > 1 and g.room2 is not None:
                            bits_aula2 = to_binary_string(g.room2, 4)
                            bits_hora2 = to_binary_string(g.start2, 5)
                        else:
                            bits_aula2 = "0000"
                            bits_hora2 = "00000"
                            
                        full_bits_sub = f"{bits_doc} {bits_dias} {bits_aula1} {bits_hora1} {bits_aula2} {bits_hora2}"
                        
                        # Cálculos Locales
                        p_sel_local = stat['aptitud'] / local_fit_sum if local_fit_sum > 0 else 0.0
                        p_acum_local += p_sel_local
                        
                        html += f"""<tr>
                        <td>{horas} Hrs</td><td>{off.cod_curso}</td><td>{off.grupo_horario}</td>
                        <td>{doc_code}</td>
                        <td style='font-family:monospace; font-size: 0.85em;'>{full_bits_sub}</td>
                        <td>{int(stat['penal'])}</td>
                        <td>{stat['aptitud']:.6f}</td>
                        <td>{local_fit_sum:.2f}</td> <td>{p_sel_local:.6f}</td><td>{p_acum_local:.6f}</td>
                        </tr>"""
                    html += "</tbody></table></div>"
                    full_html += html
                
                return full_html

            st.markdown(render_subpop_table("M", "Turno Mañana"), unsafe_allow_html=True)
            st.markdown(render_subpop_table("T", "Turno Tarde"), unsafe_allow_html=True)
            st.markdown(render_subpop_table("N", "Turno Noche"), unsafe_allow_html=True)

            # 3. REPORTE DE PENALIDADES
            st.subheader("3. Reporte de Penalidades")
            c1, c2, c3 = st.columns(3)
            def make_list_html(items, empty_msg):
                content = ""
                if not items: content = f"<span style='color:green; font-weight:bold;'>{empty_msg}</span>"
                else: 
                    for item in items: content += f"{item}<br>"
                return f"<div class='penalty-list'>{content}</div>"

            with c1:
                st.markdown("**Penalidad Docente**")
                st.markdown(make_list_html(pen_doc, "✔ 0 Conflictos"), unsafe_allow_html=True)
            with c2:
                st.markdown("**Penalidad Docente Ciclo**")
                st.markdown(make_list_html(pen_ciclo, "✔ 0 Conflictos"), unsafe_allow_html=True)
            with c3:
                st.markdown("**Penalidad Docente Aula**")
                st.markdown(make_list_html(pen_aula, "✔ 0 Conflictos"), unsafe_allow_html=True)

    # 3. MATRICES DE CONFLICTOS
    elif page == "Matrices de Conflictos":
        st.header("⚠️ Matrices de Conflictos")
        
        if not st.session_state.get("has_run", False):
            st.warning("Debe inicializar los datos primero.")
        else:
            best = st.session_state.best_ind
            offers = st.session_state.offers
            m_cycle, m_room, m_teach = get_conflict_matrices(best, offers, bundle)
            
            tipo_matriz = st.radio("Tipo de Matriz:", 
                                   ["Por Ciclo (Cruce Horario)", "Por Aula (Ocupación)", "Por Docente (Disponibilidad)"], 
                                   horizontal=True)
            st.write("---")

            def style_conflict_matrix(df):
                def highlight_cells(val):
                    try:
                        v = int(val)
                        if v == 0: return 'color: #ccc;' 
                        if v == 1: return 'background-color: #c8e6c9; color: #1b5e20; font-weight: bold;' 
                        if v > 1: return 'background-color: #ffcdd2; color: #b71c1c; font-weight: bold;'
                    except: pass
                    return ''
                return df.style.applymap(highlight_cells)

            idx_horas = [horas_map.get(i, str(i)) for i in range(17)]
            cols_dias = [dias_map.get(i, str(i)) for i in range(6)]

            if "Ciclo" in tipo_matriz:
                ciclos_disponibles = sorted(list(set(o.ciclo for o in offers)))
                sel_ciclo = st.selectbox("Seleccione Ciclo:", ciclos_disponibles)
                mat_data = m_cycle[sel_ciclo - 1].T 
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                st.dataframe(style_conflict_matrix(df_viz), height=600, use_container_width=True)

            elif "Aula" in tipo_matriz:
                aulas_sorted = bundle.aulas.sort_values("cod_aula")
                aula_ids = aulas_sorted["room_id"].tolist()
                aula_dict = dict(zip(aulas_sorted["room_id"], aulas_sorted["cod_aula"]))
                sel_aula_id = st.selectbox("Seleccione Aula:", options=aula_ids, format_func=lambda x: aula_dict[x])
                mat_data = m_room[sel_aula_id].T
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                st.dataframe(style_conflict_matrix(df_viz), height=600, use_container_width=True)

            elif "Docente" in tipo_matriz:
                docentes_sorted = bundle.docentes.sort_values("ap_paterno")
                doc_ids = sorted(docentes_sorted["teacher_id"].unique())
                sel_doc_id = st.selectbox("Seleccione Docente:", options=doc_ids, format_func=lambda x: get_docente_name(bundle, x))
                mat_data = m_teach[sel_doc_id].T
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                st.dataframe(style_conflict_matrix(df_viz), height=600, use_container_width=True)

    # 3.5 EDITOR MANUAL (V4 - LIMPIO Y CON CÓDIGOS)
    elif page == "Editor Manual":
        st.header("Editor Manual de Horarios")
        st.markdown("""
        **Guía de uso:**
        1. Seleccione el Ciclo.
        2. Seleccione el curso a corregir (Ordenado por gravedad del conflicto).
        3. Mueva el horario a un hueco libre para eliminar la penalidad.
        """)

        if not st.session_state.get("has_run", False):
            st.warning("Primero ejecute 'CREAR POBLACIÓN INICIAL' en la pestaña 'Realizar Asignación'.")
        else:
            # Recuperar variables necesarias
            solver = st.session_state.solver
            best = st.session_state.best_ind
            offers = st.session_state.offers
            bundle = st.session_state.bundle
            cfg = GAConfig(N_SLOTS_PER_DAY=17)

            # 1. Recalcular estado actual
            gene_stats, _, pen_doc, pen_cyc, pen_room = calculate_dynamic_details(best, offers, bundle)
            
            # Datos auxiliares
            aulas_dict = dict(zip(bundle.aulas["room_id"], bundle.aulas["cod_aula"]))
            horas_dict = dict(zip(bundle.horas["slot_id"], bundle.horas["rango_hora"]))
            dias_list = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]

            # --- BARRA DE FILTROS ---
            col_check, _ = st.columns([2, 1])
            show_conflicts_only = col_check.checkbox("Mostrar solo cursos con conflictos activos", value=True)

            st.divider()
            
            # --- SELECCIÓN JERÁRQUICA ---
            col_ciclo, col_curso = st.columns([1, 2])

            # PASO 1: SELECCIONAR CICLO
            all_cycles = sorted(list(set(o.ciclo for o in offers)))
            sel_ciclo = col_ciclo.selectbox("1. Seleccione Ciclo:", all_cycles)

            # PASO 2: LISTA DE CURSOS (LIMPIA Y CON CÓDIGOS)
            options_data = []
            
            for stat in gene_stats:
                idx = stat['idx']
                penal = stat['penal']
                off = offers[idx]
                
                # Filtros
                if off.ciclo != sel_ciclo: continue
                if show_conflicts_only and penal == 0: continue
                
                gene = best.genes[idx]
                doc_code = get_docente_code(bundle, gene.teacher_id)
                doc_name = get_docente_name(bundle, gene.teacher_id)
                
                # Analizar criticidad para el ordenamiento (oculto al usuario)
                keyword = f"{off.cod_curso}"
                has_room_error = any(keyword in p for p in pen_room)
                has_doc_error = any((keyword in p and doc_code in p) for p in pen_doc)
                is_critical = has_room_error or has_doc_error

                # ETIQUETA LIMPIA: [C01] Nombre... | Doc: [D01] Nombre... | Pen: 10
                label = f"[{off.cod_curso}] {off.nombre} ({off.grupo_horario}) | Doc: [{doc_code}] {doc_name} | Pen: {int(penal)}"
                
                options_data.append({
                    "label": label,
                    "idx": idx,
                    "is_critical": is_critical,
                    "penal": penal
                })

            # Ordenar: Primero Críticos, luego por mayor penalidad
            options_data.sort(key=lambda x: (not x['is_critical'], -x['penal']))
            
            course_map = {item["label"]: item["idx"] for item in options_data}

            sel_idx = None
            with col_curso:
                if not course_map:
                    st.selectbox("2. Seleccione Curso:", ["(No hay conflictos en este ciclo)"], disabled=True)
                else:
                    sel_label = st.selectbox(f"2. Seleccione Curso ({len(course_map)}):", list(course_map.keys()))
                    sel_idx = course_map[sel_label]

            # --- ÁREA DE EDICIÓN ---
            if sel_idx is not None:
                idx = sel_idx
                gene = best.genes[idx]
                off = offers[idx]
                
                st.markdown("---")
                
                # === DIAGNÓSTICO (TEXTO PLANO) ===
                doc_code_actual = get_docente_code(bundle, gene.teacher_id)
                keyword = f"{off.cod_curso}"
                
                err_room = [p for p in pen_room if keyword in p]
                err_doc = [p for p in pen_doc if keyword in p and doc_code_actual in p]
                err_cyc = [p for p in pen_cyc if keyword in p]

                c_diag, c_form = st.columns([1, 2])
                
                with c_diag:
                    st.subheader("Diagnóstico")
                    if not (err_room or err_doc or err_cyc):
                        st.success("Curso sin conflictos detectados.")
                    else:
                        if err_room:
                            st.error(f"CONFLICTO DE AULA\nEl aula asignada ya está ocupada en ese horario.")
                        if err_doc:
                            st.error(f"CONFLICTO DE DOCENTE\nEl profesor {doc_code_actual} tiene cruce de horario.")
                        if err_cyc:
                            st.warning(f"CONFLICTO DE CICLO\nSe cruza con otro curso del mismo ciclo.")

                # === FORMULARIO ===
                with c_form:
                    with st.form("manual_edit_form"):
                        st.markdown(f"**Editando:** [{off.cod_curso}] {off.nombre} - {off.grupo_horario}")
                        
                        col_b1, col_b2 = st.columns(2)
                        
                        # BLOQUE 1
                        with col_b1:
                            st.info("Sesión 1")
                            new_day1 = st.selectbox("Día", range(6), index=gene.days[0], format_func=lambda x: dias_list[x], key="d1")
                            new_start1 = st.selectbox("Hora Inicio", range(17), index=gene.start1, format_func=lambda x: horas_dict.get(x, str(x)), key="h1")
                            
                            room_ids = list(aulas_dict.keys())
                            curr_room = gene.room1
                            idx_r1 = room_ids.index(curr_room) if curr_room in room_ids else 0
                            new_room1 = st.selectbox("Aula", room_ids, index=idx_r1, format_func=lambda x: aulas_dict[x], key="r1")
                            
                            st.caption(f"Duración: {gene.len1} hrs")

                        # BLOQUE 2 (Opcional)
                        new_day2, new_start2, new_room2 = None, None, None
                        if len(gene.days) > 1 and gene.room2 is not None:
                            with col_b2:
                                st.info("Sesión 2")
                                new_day2 = st.selectbox("Día", range(6), index=gene.days[1], format_func=lambda x: dias_list[x], key="d2")
                                new_start2 = st.selectbox("Hora Inicio", range(17), index=gene.start2, format_func=lambda x: horas_dict.get(x, str(x)), key="h2")
                                
                                curr_room2 = gene.room2
                                idx_r2 = room_ids.index(curr_room2) if curr_room2 in room_ids else 0
                                new_room2 = st.selectbox("Aula", room_ids, index=idx_r2, format_func=lambda x: aulas_dict[x], key="r2")
                                
                                st.caption(f"Duración: {gene.len2} hrs")

                        st.write("")
                        btn_save = st.form_submit_button("GUARDAR SOLUCIÓN Y RECALCULAR", type="primary")
                        
                        if btn_save:
                            # Aplicar cambios en memoria
                            dias_nuevos = [new_day1]
                            if new_day2 is not None: dias_nuevos.append(new_day2)
                            
                            gene.days = tuple(dias_nuevos)
                            gene.room1 = new_room1
                            gene.start1 = new_start1
                            
                            if new_day2 is not None:
                                gene.room2 = new_room2
                                gene.start2 = new_start2
                            
                            # Recalcular penalidad global
                            evaluate(best, offers, solver.n_cycles, solver.n_rooms, solver.n_teachers, cfg)
                            st.session_state.best_ind = best
                            
                            st.success(f"¡Guardado! Nueva Penalidad Global: {int(best.penalty)}")
                            st.rerun()

    # 4. HORARIO POR CICLO
    elif page == "Horario por Ciclo":
        st.header("📅 Horario por Ciclo Académico")
        if not st.session_state.get("has_run", False):
            st.warning("Ejecute la asignación primero.")
        else:
            offers = st.session_state.offers
            best = st.session_state.best_ind
            ciclos = sorted(list(set(o.ciclo for o in offers)))
            sel_ciclo = st.selectbox("Seleccione Ciclo:", ciclos)
            if sel_ciclo:
                df_c = create_schedule_matrix(best, offers, bundle, "Ciclo", sel_ciclo, dias_map, horas_map)
                st.markdown(df_c.to_html(escape=False, classes="schedule-table"), unsafe_allow_html=True)

    # 5. HORARIO POR AULA
    elif page == "Horario por Aula":
        st.header("🏫 Horario por Aula/Ambiente")
        if not st.session_state.get("has_run", False):
            st.warning("Ejecute la asignación primero.")
        else:
            offers = st.session_state.offers
            best = st.session_state.best_ind
            aulas_sorted = bundle.aulas.sort_values("cod_aula")
            aula_ids = aulas_sorted["room_id"].tolist()
            aula_options = dict(zip(aulas_sorted["room_id"], aulas_sorted["cod_aula"]))
            sel_aula = st.selectbox("Seleccione Aula:", options=aula_ids, format_func=lambda x: aula_options[x])
            if sel_aula is not None:
                df_a = create_schedule_matrix(best, offers, bundle, "Aula", sel_aula, dias_map, horas_map)
                st.markdown(df_a.to_html(escape=False, classes="schedule-table"), unsafe_allow_html=True)

    # 6. HORARIOS EN GENERAL
    elif page == "Horarios en General":
        st.header("✅ Tabla de Resultados Finales")
        if not st.session_state.get("has_run", False):
            st.warning("Ejecute la asignación primero.")
        else:
            offers = st.session_state.offers
            best = st.session_state.best_ind
            data_final = []
            for g, off in zip(best.genes, offers):
                def add_row(d_idx, start, room, sesion):
                    data_final.append({
                        "Ciclo": off.ciclo,
                        "Cód Curso": off.cod_curso,
                        "Curso": off.nombre,
                        "Grupo": off.grupo_horario,
                        "Docente": get_docente_name(bundle, g.teacher_id),
                        "Aula": get_room_code(bundle, room),
                        "Día": dias_map.get(d_idx, str(d_idx)),
                        "Hora": horas_map.get(start, str(start)),
                        "Sesión": sesion
                    })
                add_row(g.days[0], g.start1, g.room1, "1")
                if len(g.days) > 1 and g.room2 is not None:
                      add_row(g.days[1], g.start2, g.room2, "2")
            df_res = pd.DataFrame(data_final)
            st.dataframe(df_res, use_container_width=True)
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar CSV", data=csv, file_name="horario_final.csv", mime="text/csv")

if __name__ == "__main__":
    main()