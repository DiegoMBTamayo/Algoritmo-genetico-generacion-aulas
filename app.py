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
from src.initial_population import build_initial_population
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
    }
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        background-color: #262730; 
        color: #fafafa;
        border: 1px solid #444;
        margin-bottom: 0px;
    }
    .dark-table th {
        background-color: #333;
        border: 1px solid #555;
        padding: 6px;
        text-align: left;
        font-weight: bold;
        color: #fff;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .dark-table td {
        border: 1px solid #444;
        padding: 4px 6px;
        color: #ddd;
    }
    .scroll-container {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #444;
        background-color: #262730;
        margin-bottom: 15px;
    }
    .dark-info-box {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        margin-bottom: 10px;
        color: #eee;
        max-height: 150px;
        overflow-y: auto;
    }
    .section-header {
        font-family: Arial, sans-serif;
        font-weight: bold;
        color: #eee;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 5px;
    }
    .read-only-input {
        background-color: #333;
        color: #fff;
        padding: 5px;
        border: 1px solid #555;
        border-radius: 4px;
        width: 100%;
        text-align: center;
        font-weight: bold;
    }
    .schedule-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 12px;
    }
    .schedule-table th {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
        font-weight: bold;
        color: #333;
    }
    .schedule-table td {
        border: 1px solid #ddd;
        padding: 4px;
        vertical-align: top;
        background-color: #fff;
        color: #000;
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
        f"<div style='border:1px solid #999; margin-bottom:4px; font-family:sans-serif; overflow:hidden;'>"
        f"<div style='background-color:#ffffcc; color:#000; padding:2px 4px; font-size:11px; font-weight:bold; border-bottom:1px solid #ccc; text-align:left;'>"
        f"{nombre}"
        f"</div>"
        f"<div style='background-color:#fff; color:#333; padding:2px 4px; font-size:10px; text-align:left;'>"
        f"<b>{grupo}</b> {docente} {aula} <span style='float:right; color:#666;'>({tipo})</span>"
        f"</div>"
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

# --- HELPER: MATRICES DE CONFLICTO (NUEVO) ---
def get_conflict_matrices(individual, offers, bundle):
    """Retorna las 3 matrices numéricas de ocupación: [Ciclo, Aula, Docente]"""
    n_cycles = 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    n_teachers = int(bundle.docentes["teacher_id"].max()) + 1
    TOTAL_SLOTS = 17

    # Matrices de conteo (0=Libre, 1=Ocupado, >1=Conflicto)
    count_cycle = np.zeros((n_cycles + 1, 6, TOTAL_SLOTS), dtype=int)
    count_room = np.zeros((n_rooms + 1, 6, TOTAL_SLOTS), dtype=int)
    count_teach = np.zeros((n_teachers + 1, 6, TOTAL_SLOTS), dtype=int)

    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        cyc = int(off.ciclo) - 1
        # Bloque 1
        for h in range(g.start1, g.start1 + g.len1):
            if h < TOTAL_SLOTS and g.days[0] < 6:
                count_cycle[cyc, g.days[0], h] += 1
                count_room[g.room1, g.days[0], h] += 1
                count_teach[g.teacher_id, g.days[0], h] += 1
        # Bloque 2
        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < TOTAL_SLOTS and g.days[1] < 6:
                    count_cycle[cyc, g.days[1], h] += 1
                    count_room[g.room2, g.days[1], h] += 1
                    count_teach[g.teacher_id, g.days[1], h] += 1
    
    return count_cycle, count_room, count_teach

# --- HELPER: CÁLCULO DINÁMICO DE PENALIDADES ---
def calculate_dynamic_details(individual, offers, bundle):
    # Reutilizamos la lógica de matrices para calcular penalidades
    count_cycle, count_room, count_teach = get_conflict_matrices(individual, offers, bundle)
    
    cfg = GAConfig()
    TOTAL_SLOTS = 17
    gene_stats = []
    penalties_docente = []
    penalties_doc_ciclo = []
    penalties_doc_aula = []
    total_fitness_sum = 0
    
    # Recorrer de nuevo para asignar culpas a cada gen
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
        total_fitness_sum += aptitud
        
        gene_stats.append({"idx": i, "penal": pen_gen, "aptitud": aptitud})
        
        doc_code = get_docente_code(bundle, g.teacher_id)
        if pen_teacher > 0: penalties_docente.append(f"{off.cod_curso} {doc_code}: {pen_teacher}")
        if pen_cycle > 0: penalties_doc_ciclo.append(f"{off.cod_curso} (Ciclo {off.ciclo}): {pen_cycle}")
        if pen_room > 0: penalties_doc_aula.append(f"{off.cod_curso} {doc_code}: {pen_room}")

    return gene_stats, total_fitness_sum, penalties_docente, penalties_doc_ciclo, penalties_doc_aula

# --- LOGICA DE EVOLUCIÓN UN PASO ---
def run_single_generation(solver, population, mutation_rate, offers, cfg):
    best_global = max(population, key=lambda x: x.fitness)
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_pop = [copy.deepcopy(best_global)]
    while len(new_pop) < len(population):
        p1 = solver.selection_roulette(population)
        p2 = solver.selection_roulette(population)
        child = solver.crossover_subpopulations(p1, p2)
        solver.mutate(child, mutation_rate)
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

    # INICIALIZACIÓN DE ESTADO
    if 'generation_count' not in st.session_state: st.session_state.generation_count = 0
    if 'history' not in st.session_state: st.session_state.history = []
    if 'population' not in st.session_state: st.session_state.population = None
    if 'best_ind' not in st.session_state: st.session_state.best_ind = None
    if 'initial_pop_list' not in st.session_state: st.session_state.initial_pop_list = None 

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.title("🧬 Menú Principal")
        st.markdown("---")
        page = st.radio("Ir a la sección:", [
            "Realizar Asignación", 
            "Procesos Adicionales", 
            "Matrices de Conflictos", # NUEVA PESTAÑA
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
        if st.button("🚀 INICIAR PROCESO COMPLETO (Reset)"):
            st.session_state.population = None
            st.session_state.generation_count = 0
            st.session_state.history = []
            
            with st.status("Iniciando...", expanded=True) as status:
                offers = build_offers(bundle)
                baseline = build_baseline_lab(bundle)
                aulas_calc = bundle.aulas.copy()
                aulas_calc["es_laboratorio"] = aulas_calc["cod_tipo"].astype(str).str.startswith("L")
                domains = build_offer_domains(offers, aulas_calc, bundle.docentes, TURN_SLOTS, 6, baseline)
                
                initial_pop = build_initial_population(offers, domains, 15, 102)
                
                n_cycles = 10
                n_rooms = int(bundle.aulas["room_id"].max()) + 1
                n_teachers = int(bundle.docentes["teacher_id"].max()) + 1
                cfg = GAConfig()
                for ind in initial_pop:
                    evaluate(ind, offers, n_cycles, n_rooms, n_teachers, cfg)
                
                st.session_state.initial_pop_list = initial_pop
                st.session_state.population = initial_pop
                st.session_state.offers = offers
                st.session_state.domains = domains
                st.session_state.solver = GeneticSolver(offers, domains, cfg, n_cycles, n_rooms, n_teachers)
                
                best = max(initial_pop, key=lambda x: x.fitness)
                st.session_state.best_ind = best
                st.session_state.history.append((0, int(best.penalty)))
                st.session_state.has_run = True
                
                status.update(label="¡Inicializado! Ve a 'Procesos Adicionales'.", state="complete", expanded=False)

        if st.session_state.get("has_run", False) and st.session_state.initial_pop_list:
            st.divider()
            st.subheader("Muestra de Población Inicial (Aleatoria)")
            st.caption("A continuación se muestran los cromosomas generados en la población inicial.")
            sample_ind = st.session_state.initial_pop_list[0]
            offers = st.session_state.offers
            pop_data = []
            for g, off in zip(sample_ind.genes, offers):
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                bits_a1 = to_binary_string(g.room1, 4)
                bits_h1 = to_binary_string(g.start1, 5)
                if len(g.days) > 1 and g.room2 is not None:
                    bits_a2 = to_binary_string(g.room2, 4)
                    bits_h2 = to_binary_string(g.start2, 5)
                else:
                    bits_a2 = "0000"
                    bits_h2 = "00000"
                pop_data.append({
                    "Curso": off.cod_curso,
                    "Grupo": off.grupo_horario,
                    "Doc(6b)": bits_doc,
                    "Días(6b)": bits_dias,
                    "Aula1(4b)": bits_a1,
                    "Hora1(5b)": bits_h1,
                    "Aula2(4b)": bits_a2,
                    "Hora2(5b)": bits_h2
                })
            st.dataframe(pd.DataFrame(pop_data), height=400, use_container_width=True)

    # 2. PROCESOS ADICIONALES
    elif page == "Procesos Adicionales":
        st.header("⚙️ Procesos Adicionales")
        
        if not st.session_state.get("has_run", False):
            st.warning("Debe inicializar los datos en la pestaña 'Realizar Asignación' primero.")
        else:
            solver = st.session_state.solver
            offers = st.session_state.offers
            cfg = GAConfig()

            col_controls, col_metrics = st.columns([2, 1])
            with col_controls:
                c1, c2, c3 = st.columns(3)
                if c1.button("Altera Poblac. Inicial"):
                    with st.spinner("Reiniciando población..."):
                        pop_size = 102
                        new_pop = build_initial_population(offers, solver.domains, 15, pop_size)
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

                if c2.button("Calcula Penalidad"):
                    for ind in st.session_state.population:
                        evaluate(ind, offers, solver.n_cycles, solver.n_rooms, solver.n_teachers, cfg)
                    best = max(st.session_state.population, key=lambda x: x.fitness)
                    st.session_state.best_ind = best
                    st.rerun()

                c4, c5, c6 = st.columns(3)
                if c4.button("Gen. UNA Población"):
                    current_pop = st.session_state.population
                    next_gen = run_single_generation(solver, current_pop, 0.1, offers, cfg)
                    st.session_state.population = next_gen
                    st.session_state.generation_count += 1
                    best = max(next_gen, key=lambda x: x.fitness)
                    st.session_state.best_ind = best
                    st.session_state.history.append((st.session_state.generation_count, int(best.penalty)))
                    st.rerun()
            
                if c5.button("Gen. Poblaciones (5)"):
                    current_pop = st.session_state.population
                    for _ in range(5):
                        current_pop = run_single_generation(solver, current_pop, 0.1, offers, cfg)
                        st.session_state.generation_count += 1
                        best = max(current_pop, key=lambda x: x.fitness)
                        st.session_state.history.append((st.session_state.generation_count, int(best.penalty)))
                    st.session_state.population = current_pop
                    st.session_state.best_ind = best
                    st.rerun()

                if c6.button("Salir"):
                    st.session_state.clear()
                    st.rerun()
            
            with col_metrics:
                gen_num = st.session_state.get('generation_count', 0)
                best_pen = int(st.session_state.best_ind.penalty) if st.session_state.best_ind else 0
                st.markdown(f"""
                <div style='display:flex; gap:10px; margin-top:5px;'>
                    <div><small>Población (Gen)</small><div class='read-only-input'>{gen_num:03d}</div></div>
                    <div><small>T.Penalidad</small><div class='read-only-input' style='color:#ff4b4b'>{best_pen}</div></div>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")
            best = st.session_state.best_ind
            gene_stats, total_fit_sum, pen_doc, pen_ciclo, pen_aula = calculate_dynamic_details(best, offers, bundle)

            st.markdown("##### Cursos Docentes (Detalle Cromosoma)")
            html_main = """<table class="dark-table"><thead><tr>
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
                bits_bloque1 = f"{to_binary_string(g.room1, 4)}{to_binary_string(g.start1, 5)}"
                bits_bloque2 = f" {to_binary_string(g.room2, 4)}{to_binary_string(g.start2, 5)}" if (len(g.days) > 1 and g.room2 is not None) else " 000000000"
                full_bits = f"{bits_doc} {bits_dias} {bits_bloque1}{bits_bloque2}"
                p_selec = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                p_acum += p_selec
                html_main += f"""<tr><td>{off.cod_curso}</td><td>{off.grupo_horario}</td><td>{doc_code}</td><td style='font-family:monospace'>{full_bits}</td><td>{int(stat['penal'])}</td><td>{stat['aptitud']:.5f}</td><td>{p_selec:.5f}</td><td>{p_acum:.5f}</td></tr>"""
            html_main += "</tbody></table>"
            st.markdown(f"<div class='scroll-container'>{html_main}</div>", unsafe_allow_html=True)

            st.markdown("##### Subpoblaciones (Turnos)")
            def render_unified_turn_table(turno_key, title):
                if turno_key not in solver.subpopulations: return ""
                html_sub = f"""<div style="margin-bottom:5px; font-weight:bold; color:#ddd;">{title}</div>
                <table class="dark-table"><thead><tr><th>Carga</th><th>Curso</th><th>Grupo</th><th>Docente</th><th>Cromosoma (Bits)</th><th>P.Sel</th></tr></thead><tbody>"""
                has_data = False
                for horas, idxs in solver.subpopulations[turno_key].items():
                    for idx in idxs:
                        has_data = True
                        stat = next((s for s in gene_stats if s['idx'] == idx), None)
                        if not stat: continue
                        g = best.genes[idx]
                        off = offers[idx]
                        doc_code = get_docente_code(bundle, g.teacher_id)
                        bits_doc = to_binary_string(g.teacher_id, 6)
                        bits_dias = day_to_bitmask(g.days)
                        bits = f"{bits_doc} {bits_dias}..."
                        p_sel = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                        html_sub += f"""<tr><td>{horas} Hrs</td><td>{off.cod_curso}</td><td>{off.grupo_horario}</td><td>{doc_code}</td><td style='font-family:monospace'>{bits}</td><td>{p_sel:.5f}</td></tr>"""
                html_sub += "</tbody></table>"
                return html_sub if has_data else ""

            full_subpop = ""
            full_subpop += render_unified_turn_table("M", "► Turno Mañana") + "<br>"
            full_subpop += render_unified_turn_table("T", "► Turno Tarde") + "<br>"
            full_subpop += render_unified_turn_table("N", "► Turno Noche")
            st.markdown(f"<div class='scroll-container' style='max_height:300px;'>{full_subpop}</div>", unsafe_allow_html=True)

            st.markdown("##### Reporte de Penalidades")
            c1, c2, c3 = st.columns(3)
            def make_list_html(items, empty_msg):
                if not items: return f"<div class='dark-info-box' style='color:#88dd88;'>{empty_msg}</div>"
                html = "<div class='dark-info-box'>"
                for item in items: html += f"{item}<br>"
                html += "</div>"
                return html
            with c1:
                st.markdown("**Penalidad Docente**")
                st.markdown(make_list_html(pen_doc, "OK"), unsafe_allow_html=True)
            with c2:
                st.markdown("**Penalidad Docente Ciclo**")
                st.markdown(make_list_html(pen_ciclo, "OK"), unsafe_allow_html=True)
            with c3:
                st.markdown("**Penalidad Docente Aula**")
                st.markdown(make_list_html(pen_aula, "OK"), unsafe_allow_html=True)

    # 3. MATRICES DE CONFLICTOS (NUEVA PESTAÑA)
    elif page == "Matrices de Conflictos":
        st.header("⚠️ Matrices de Conflictos")
        
        if not st.session_state.get("has_run", False):
            st.warning("Debe inicializar los datos en la pestaña 'Realizar Asignación' primero.")
        else:
            best = st.session_state.best_ind
            offers = st.session_state.offers
            # Obtenemos las matrices crudas
            m_cycle, m_room, m_teach = get_conflict_matrices(best, offers, bundle)
            
            # Selector de Tipo
            tipo_matriz = st.radio("Seleccione el Tipo de Matriz:", 
                                   ["Por Ciclo (Cruce de Horarios)", "Por Aula (Ocupación)", "Por Docente (Disponibilidad)"], 
                                   horizontal=True)
            st.write("---")

            def style_conflict_matrix(df):
                """Aplica colores: 0=Vacío, 1=Verde (OK), >1=Rojo (Conflicto)"""
                def highlight_cells(val):
                    try:
                        v = int(val)
                        if v == 0: return 'color: #333;' # Ocultar ceros (vacío)
                        if v == 1: return 'background-color: #90ee90; color: black; font-weight: bold;' # Verde
                        if v > 1: return 'background-color: #ff4b4b; color: white; font-weight: bold;' # Rojo
                    except: pass
                    return ''
                return df.style.applymap(highlight_cells)

            # Preparar índices y columnas para visualización
            idx_horas = [horas_map.get(i, str(i)) for i in range(17)]
            cols_dias = [dias_map.get(i, str(i)) for i in range(6)]

            if "Ciclo" in tipo_matriz:
                # Selector de ciclo
                ciclos_disponibles = sorted(list(set(o.ciclo for o in offers)))
                sel_ciclo = st.selectbox("Seleccione Ciclo:", ciclos_disponibles)
                
                # Obtener la matriz específica (cycle_idx = ciclo - 1)
                # m_cycle shape: (cycles, days, slots) -> Transponemos a (slots, days)
                mat_data = m_cycle[sel_ciclo - 1].T 
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                
                st.subheader(f"Matriz de Conflictos: Ciclo {sel_ciclo}")
                st.dataframe(style_conflict_matrix(df_viz), height=600)

            elif "Aula" in tipo_matriz:
                # Selector de Aula (Ordenado alfabéticamente)
                aulas_sorted = bundle.aulas.sort_values("cod_aula")
                aula_ids = aulas_sorted["room_id"].tolist()
                aula_dict = dict(zip(aulas_sorted["room_id"], aulas_sorted["cod_aula"]))
                
                sel_aula_id = st.selectbox("Seleccione Aula:", options=aula_ids, format_func=lambda x: aula_dict[x])
                
                mat_data = m_room[sel_aula_id].T
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                
                st.subheader(f"Matriz de Conflictos: {aula_dict[sel_aula_id]}")
                st.dataframe(style_conflict_matrix(df_viz), height=600)

            elif "Docente" in tipo_matriz:
                # Selector de Docente
                docentes_sorted = bundle.docentes.sort_values("ap_paterno")
                # Filtramos solo docentes que están en el bundle
                doc_ids = sorted(docentes_sorted["teacher_id"].unique())
                
                def get_doc_label(tid):
                    return get_docente_name(bundle, tid)
                
                sel_doc_id = st.selectbox("Seleccione Docente:", options=doc_ids, format_func=get_doc_label)
                
                mat_data = m_teach[sel_doc_id].T
                df_viz = pd.DataFrame(mat_data, index=idx_horas, columns=cols_dias)
                
                st.subheader(f"Matriz de Conflictos: {get_docente_name(bundle, sel_doc_id)}")
                st.dataframe(style_conflict_matrix(df_viz), height=600)

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