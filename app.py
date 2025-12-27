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
# Usamos la población inicial estándar (Probabilística/Aleatoria)
from src.initial_population import build_initial_population, random_gene_for_offer
from src.ga import GeneticSolver
from src.evaluation import evaluate
from run import build_offers, build_baseline_lab, to_binary_string, day_to_bitmask

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema de Horarios FIIS", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS ---
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
        font-size: 0.85em;
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
        padding: 8px 12px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
        background-color: #ffffff;
        color: #333;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    .scroll-table-container {
        height: 300px;
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
        background-color: #eee;
        padding: 8px;
        font-weight: bold;
        color: #333;
        border-left: 5px solid #009879;
        margin-top: 15px;
        margin-bottom: 5px;
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

# --- HELPER: CÁLCULO DINÁMICO DE PENALIDADES ---
def calculate_dynamic_details(individual, offers, bundle):
    count_cycle, count_room, count_teach = get_conflict_matrices(individual, offers, bundle)
    cfg = GAConfig()
    TOTAL_SLOTS = 17
    
    gene_stats = []
    penalties_docente = []
    penalties_doc_ciclo = []
    penalties_doc_aula = []
    
    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        pen_gen = 0
        pen_teacher = 0
        pen_cycle = 0
        pen_room = 0
        cyc = int(off.ciclo) - 1
        
        # Bloque 1
        for h in range(g.start1, g.start1 + g.len1):
            if h < TOTAL_SLOTS and g.days[0] < 6:
                if count_cycle[cyc, g.days[0], h] > 1: pen_cycle += cfg.W_CONFLICT_CYCLE
                if count_room[g.room1, g.days[0], h] > 1: pen_room += cfg.W_CONFLICT_ROOM
                if count_teach[g.teacher_id, g.days[0], h] > 1: pen_teacher += cfg.W_CONFLICT_TEACHER

        # Bloque 2
        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < TOTAL_SLOTS and g.days[1] < 6:
                    if count_cycle[cyc, g.days[1], h] > 1: pen_cycle += cfg.W_CONFLICT_CYCLE
                    if count_room[g.room2, g.days[1], h] > 1: pen_room += cfg.W_CONFLICT_ROOM
                    if count_teach[g.teacher_id, g.days[1], h] > 1: pen_teacher += cfg.W_CONFLICT_TEACHER
        
        pen_gen = pen_cycle + pen_room + pen_teacher
        aptitud = 1.0 / (1.0 + pen_gen)
        
        gene_stats.append({"idx": i, "penal": pen_gen, "aptitud": aptitud})
        
        doc_code = get_docente_code(bundle, g.teacher_id)
        if pen_teacher > 0: penalties_docente.append(f"{off.cod_curso} {doc_code}: {pen_teacher}")
        if pen_cycle > 0: penalties_doc_ciclo.append(f"{off.cod_curso} (Ciclo {off.ciclo}): {pen_cycle}")
        if pen_room > 0: penalties_doc_aula.append(f"{off.cod_curso} {doc_code}: {pen_room}")

    return gene_stats, 0, penalties_docente, penalties_doc_ciclo, penalties_doc_aula

# --- LOGICA DE EVOLUCIÓN OPTIMIZADA (TORNEO) ---
def tournament_selection(population, k=5):
    candidates = random.sample(population, k)
    return max(candidates, key=lambda x: x.fitness)

def smart_mutate(individual, offers, domains, mutation_rate, cfg):
    if random.random() > mutation_rate:
        return

    idx = random.randint(0, len(individual.genes) - 1)
    if individual.genes[idx].frozen: return

    # Mutación simple (aleatoria) para mantener diversidad y no caer en mínimos locales
    new_gene = random_gene_for_offer(offers[idx], domains[idx], cfg.N_SLOTS_PER_DAY)
    individual.genes[idx] = new_gene

def run_smart_generation(solver, population, mutation_rate, offers, cfg):
    pop_size = len(population)
    elite_count = int(pop_size * 0.20)
    if elite_count < 2: elite_count = 2
    
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_pop = [copy.deepcopy(ind) for ind in population[:elite_count]]
    
    while len(new_pop) < pop_size:
        p1 = tournament_selection(population, k=5)
        p2 = tournament_selection(population, k=5)
        
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

    # Forzamos 17 slots para evitar errores de índice en turno noche
    APP_CONFIG = GAConfig(N_SLOTS_PER_DAY=17) 

    with st.sidebar:
        st.title("🧬 Menú Principal")
        st.markdown("---")
        page = st.radio("Ir a la sección:", [
            "Realizar Asignación", 
            "Procesos Adicionales", 
            "Matrices de Conflictos",
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
                
                # USAMOS LA INICIALIZACIÓN ESTÁNDAR (ALEATORIA)
                initial_pop = build_initial_population(offers, domains, APP_CONFIG.N_SLOTS_PER_DAY, 102)
                
                n_cycles = 10
                n_rooms = int(bundle.aulas["room_id"].max()) + 1
                n_teachers = int(bundle.docentes["teacher_id"].max()) + 1
                
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
                        # Usar build_initial_population (Aleatoria)
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
            gene_stats, total_fit_sum, pen_doc, pen_ciclo, pen_aula = calculate_dynamic_details(best, offers, bundle)

            # 1. DETALLE POR CURSO
            st.subheader("1. Detalle por curso")
            html_main = """<table class="styled-table"><thead><tr>
            <th>Curso</th><th>Grupo</th><th>Docente</th><th>Cromosoma (Docente | Días | Aula | Hora)</th><th>Penal</th><th>Aptitud</th><th>P.Selec</th><th>P.Acumul</th>
            </tr></thead><tbody>"""
            p_acum = 0.0
            for stat in gene_stats:
                idx = stat['idx']
                g = best.genes[idx]
                off = offers[idx]
                doc_code = get_docente_code(bundle, g.teacher_id)
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                bits_b1 = f"{to_binary_string(g.room1, 4)}{to_binary_string(g.start1, 5)}"
                bits_b2 = f" {to_binary_string(g.room2, 4)}{to_binary_string(g.start2, 5)}" if (len(g.days)>1 and g.room2 is not None) else " 000000000"
                full_bits = f"{bits_doc} {bits_dias} {bits_b1}{bits_b2}"
                p_selec = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                p_acum += p_selec
                html_main += f"""<tr><td>{off.cod_curso}</td><td>{off.grupo_horario}</td><td>{doc_code}</td><td style='font-family:monospace'>{full_bits}</td><td>{int(stat['penal'])}</td><td>{stat['aptitud']:.5f}</td><td>{p_selec:.5f}</td><td>{p_acum:.5f}</td></tr>"""
            html_main += "</tbody></table>"
            st.markdown(f"<div class='scroll-table-container'>{html_main}</div>", unsafe_allow_html=True)

            # 2. SUBPOBLACIONES
            st.subheader("2. Subpoblaciones")
            
            def render_subpop_table(turno_key, title):
                # Si no hay datos, retornar placeholder vacío
                if turno_key not in solver.subpopulations:
                    return ""
                
                # HTML para la tabla del bloque específico
                html = f"""
                <div class="subpop-header">{title}</div>
                <div class="scroll-table-container" style="height:200px;">
                <table class="styled-table" style="margin-top:0;">
                <thead><tr><th>Carga</th><th>Curso</th><th>Grupo</th><th>Docente</th><th>Cromosoma (Bits)</th><th>P.Sel</th></tr></thead>
                <tbody>
                """
                
                has_rows = False
                for horas, idxs in solver.subpopulations[turno_key].items():
                    for idx in idxs:
                        has_rows = True
                        stat = next((s for s in gene_stats if s['idx'] == idx), None)
                        if not stat: continue
                        g = best.genes[idx]
                        off = offers[idx]
                        doc_code = get_docente_code(bundle, g.teacher_id)
                        bits = f"{to_binary_string(g.teacher_id, 6)} {day_to_bitmask(g.days)}..."
                        p_sel = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                        
                        html += f"""<tr>
                        <td>{horas} Hrs</td><td>{off.cod_curso}</td><td>{off.grupo_horario}</td>
                        <td>{doc_code}</td><td style='font-family:monospace'>{bits}</td><td>{p_sel:.5f}</td>
                        </tr>"""
                
                html += "</tbody></table></div>"
                return html if has_rows else f"<div class='subpop-header'>{title} (Sin datos)</div>"

            # Renderizar los 3 bloques
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