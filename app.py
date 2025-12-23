# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_loader import load_data
from src.config import GAConfig, TURN_SLOTS
from src.domains import build_offer_domains
from src.initial_population import build_initial_population
from src.ga import GeneticSolver
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
        padding: 10px;
    }
    
    /* Estilos para tablas oscuras */
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        background-color: #262730; 
        color: #fafafa;
        border: 1px solid #444;
        margin-bottom: 15px;
    }
    .dark-table th {
        background-color: #333;
        border: 1px solid #555;
        padding: 6px;
        text-align: left;
        font-weight: bold;
        color: #fff;
    }
    .dark-table td {
        border: 1px solid #444;
        padding: 4px 6px;
        color: #ddd;
    }
    
    .dark-info-box {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        margin-bottom: 10px;
        color: #eee;
        max-height: 250px;
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
    if "tid_num" not in bundle.docentes.columns:
        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    row = bundle.docentes[bundle.docentes["tid_num"] == teacher_id]
    if not row.empty:
        r = row.iloc[0]
        return f"{r['nombres']} {r['ap_paterno']}"
    return f"Docente {teacher_id}"

def get_docente_code(bundle, teacher_id):
    if "tid_num" not in bundle.docentes.columns:
        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    row = bundle.docentes[bundle.docentes["tid_num"] == teacher_id]
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
    # AJUSTE: Matriz de 17 filas para cubrir hasta 22:10 (Indices 0 a 16)
    TOTAL_SLOTS = 17 
    matrix = np.full((TOTAL_SLOTS, 6), "", dtype=object)
    
    for g, off in zip(best_ind.genes, offers):
        show = False
        if filter_mode == "Ciclo" and str(off.ciclo) == str(filter_val): show = True
        if filter_mode == "Aula" and (g.room1 == filter_val or g.room2 == filter_val): show = True
        
        if not show: continue

        doc_code = get_docente_code(bundle, g.teacher_id)
        room1_code = get_room_code(bundle, g.room1)
        
        # Bloque 1
        if filter_mode == "Ciclo" or g.room1 == filter_val:
            for d in [g.days[0]]:
                if 0 <= d < 6:
                    for h in range(g.start1, g.start1 + g.len1):
                        if 0 <= h < TOTAL_SLOTS:
                            matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room1_code, off.tipo_hora)
        
        # Bloque 2
        if len(g.days) > 1 and g.room2 is not None:
            if filter_mode == "Ciclo" or g.room2 == filter_val:
                room2_code = get_room_code(bundle, g.room2)
                for d in [g.days[1]]:
                    if 0 <= d < 6:
                        for h in range(g.start2, g.start2 + g.len2):
                            if 0 <= h < TOTAL_SLOTS:
                                matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room2_code, off.tipo_hora)
    
    # Usar el mapa dinámico para las columnas
    cols = [dias_map.get(i, f"Dia {i}") for i in range(6)]
    df_mat = pd.DataFrame(matrix, columns=cols)
    # Usar el mapa dinámico para el índice (horas)
    # Aseguramos que cubra hasta 17 slots
    df_mat.index = [horas_map.get(i, f"Slot {i}") for i in range(TOTAL_SLOTS)]
    return df_mat

# --- HELPER: CÁLCULO DINÁMICO DE PENALIDADES ---
def calculate_dynamic_details(individual, offers, bundle):
    cfg = GAConfig()
    n_cycles = 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    if "tid_num" not in bundle.docentes.columns:
        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    n_teachers = int(bundle.docentes["tid_num"].max()) + 1

    # Ajustado a 17 slots
    TOTAL_SLOTS = 17
    count_cycle = np.zeros((n_cycles + 1, 6, TOTAL_SLOTS), dtype=int)
    count_room = np.zeros((n_rooms + 1, 6, TOTAL_SLOTS), dtype=int)
    count_teach = np.zeros((n_teachers + 1, 6, TOTAL_SLOTS), dtype=int)

    # 1. Llenar ocupación
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

    gene_stats = []
    penalties_docente = []
    penalties_doc_ciclo = []
    penalties_doc_aula = []
    total_fitness_sum = 0
    
    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        pen_gen = 0
        pen_teacher = 0
        pen_cycle = 0
        pen_room = 0
        
        cyc = int(off.ciclo) - 1
        
        for h in range(g.start1, g.start1 + g.len1):
            if h < TOTAL_SLOTS and g.days[0] < 6:
                if count_cycle[cyc, g.days[0], h] > 1: pen_cycle += cfg.W_CONFLICT_CYCLE
                if count_room[g.room1, g.days[0], h] > 1: pen_room += cfg.W_CONFLICT_ROOM
                if count_teach[g.teacher_id, g.days[0], h] > 1: pen_teacher += cfg.W_CONFLICT_TEACHER

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

# --- MAIN APP ---
def main():
    # 1. Cargar Datos
    if 'bundle' not in st.session_state:
        st.session_state.bundle = load_data("data")
    bundle = st.session_state.bundle

    # 2. Generar Mapeos Dinámicos
    dias_df = bundle.dias
    dias_map = dict(zip(dias_df['day_idx'], dias_df['dia']))
    
    horas_df = bundle.horas
    horas_map = dict(zip(horas_df['slot_id'], horas_df['rango_hora']))

    # --- BARRA LATERAL ---
    with st.sidebar:
        st.title("🧬 Menú Principal")
        st.markdown("---")
        page = st.radio("Ir a la sección:", [
            "Realizar Asignación", 
            "Procesos Adicionales", 
            "Horario por Ciclo", 
            "Horario por Aula", 
            "Horarios en General"
        ])
        st.markdown("---")
        st.info("Sistema de Optimización de Horarios\nAlgoritmo Genético")

    # 1. REALIZAR ASIGNACIÓN (CON CRUD)
    if page == "Realizar Asignación":
        st.header("📋 Realizar Asignación y Gestión de Datos")
        
        # --- AVISO DE GUARDADO CORRECTO ---
        # Verificamos si existe la bandera en session state
        if st.session_state.get('show_save_success', False):
            st.success("¡Datos actualizados y recargados correctamente!")
            # Apagamos la bandera para que no salga siempre
            st.session_state['show_save_success'] = False

        st.caption("Puede editar, agregar o eliminar registros en las tablas. Recuerde GUARDAR los cambios.")

        # Contenedor para las tablas editables
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

        # BOTÓN PARA GUARDAR CAMBIOS EN CSV
        if st.button("💾 Guardar Cambios en Archivos CSV"):
            try:
                edited_docentes.to_csv("data/docentes.csv", index=False)
                edited_aulas.to_csv("data/aulas.csv", index=False)
                edited_cursos.to_csv("data/cursos.csv", index=False)
                edited_fija.to_csv("data/clase_fija.csv", index=False)
                
                # Recargar el bundle
                st.session_state.bundle = load_data("data")
                # Activar bandera de éxito
                st.session_state['show_save_success'] = True
                st.rerun()
            except Exception as e:
                st.error(f"Error al guardar archivos: {e}")

        st.divider()

        # BOTÓN DE EJECUCIÓN
        if st.button("🚀 CREAR POBLACIÓN INICIAL"):
            with st.status("Ejecutando proceso...", expanded=True) as status:
                st.write("Generando ofertas...")
                offers = build_offers(bundle)
                baseline = build_baseline_lab(bundle)
                aulas_calc = bundle.aulas.copy()
                aulas_calc["es_laboratorio"] = aulas_calc["cod_tipo"].astype(str).str.startswith("L")
                
                domains = build_offer_domains(offers, aulas_calc, bundle.docentes, TURN_SLOTS, 6, baseline)
                # Params fijos
                pop_size = 102
                generations = 100
                mutation_rate = 0.1
                
                initial_pop = build_initial_population(offers, domains, 15, pop_size)
                st.session_state.initial_pop_list = initial_pop 
                
                n_cycles = 10
                n_rooms = int(bundle.aulas["room_id"].max()) + 1
                if "tid_num" not in bundle.docentes.columns:
                    bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
                n_teachers = int(bundle.docentes["tid_num"].max()) + 1
                
                cfg = GAConfig()
                solver = GeneticSolver(offers, domains, cfg, n_cycles, n_rooms, n_teachers)
                
                st.write("Optimizando horarios (Evolución)...")
                best_ind = solver.evolve(initial_pop, generations, mutation_rate)
                
                st.session_state.best_ind = best_ind
                st.session_state.offers = offers
                st.session_state.solver = solver
                st.session_state.has_run = True
                status.update(label="¡Población Generada y Optimizada!", state="complete", expanded=False)

        # MUESTRA POBLACIÓN INICIAL
        if st.session_state.get("has_run", False):
            st.divider()
            st.subheader("3. Muestra de Población Inicial (Aleatoria)")
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
        
        # Botón para re-generar
        if st.button("🔄 Generar Nueva Población"):
            if 'bundle' in st.session_state:
                with st.status("Re-generando optimización...", expanded=True) as status:
                    offers = build_offers(bundle)
                    baseline = build_baseline_lab(bundle)
                    aulas_calc = bundle.aulas.copy()
                    aulas_calc["es_laboratorio"] = aulas_calc["cod_tipo"].astype(str).str.startswith("L")
                    domains = build_offer_domains(offers, aulas_calc, bundle.docentes, TURN_SLOTS, 6, baseline)
                    pop_size, generations, mutation_rate = 102, 100, 0.1
                    initial_pop = build_initial_population(offers, domains, 15, pop_size)
                    st.session_state.initial_pop_list = initial_pop 
                    n_cycles = 10
                    n_rooms = int(bundle.aulas["room_id"].max()) + 1
                    if "tid_num" not in bundle.docentes.columns:
                        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
                    n_teachers = int(bundle.docentes["tid_num"].max()) + 1
                    cfg = GAConfig()
                    solver = GeneticSolver(offers, domains, cfg, n_cycles, n_rooms, n_teachers)
                    best_ind = solver.evolve(initial_pop, generations, mutation_rate)
                    st.session_state.best_ind = best_ind
                    st.session_state.offers = offers
                    st.session_state.solver = solver
                    st.session_state.has_run = True
                    status.update(label="¡Nueva Población Lista!", state="complete", expanded=False)
                    st.rerun()

        if not st.session_state.get("has_run", False):
            st.warning("No hay datos generados. Haga clic en 'Generar Nueva Población' arriba.")
        else:
            solver = st.session_state.solver
            best = st.session_state.best_ind
            offers = st.session_state.offers

            gene_stats, total_fit_sum, pen_doc, pen_ciclo, pen_aula = calculate_dynamic_details(best, offers, bundle)

            st.markdown("<div class='section-header'>Población y Probabilidades de Selección (Detalle Global)</div>", unsafe_allow_html=True)
            
            html_main = """<table class="dark-table">
<thead>
<tr>
<th>Curso</th>
<th>Grupo</th>
<th>Docente</th>
<th>Cromosoma (Docente | Días | Aula | Hora)</th>
<th>Penal</th>
<th>Aptitud</th>
<th>P.Selec</th>
<th>P.Acumul</th>
</tr>
</thead>
<tbody>"""
            p_acum = 0.0
            for stat in gene_stats[:50]:
                idx = stat['idx']
                g = best.genes[idx]
                off = offers[idx]
                doc_code = get_docente_code(bundle, g.teacher_id)
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                bits_bloque1 = f"{to_binary_string(g.room1, 4)}{to_binary_string(g.start1, 5)}"
                if len(g.days) > 1 and g.room2 is not None:
                    bits_bloque2 = f" {to_binary_string(g.room2, 4)}{to_binary_string(g.start2, 5)}"
                else:
                    bits_bloque2 = " 000000000"
                full_bits = f"{bits_doc} {bits_dias} {bits_bloque1}{bits_bloque2}"
                p_selec = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                p_acum += p_selec
                
                html_main += f"""<tr>
<td>{off.cod_curso}</td>
<td>{off.grupo_horario}</td>
<td>{doc_code}</td>
<td>{full_bits}</td>
<td>{int(stat['penal'])}</td>
<td>{stat['aptitud']:.6f}</td>
<td>{p_selec:.6f}</td>
<td>{p_acum:.6f}</td>
</tr>"""
            html_main += "</tbody></table>"
            st.markdown(f"<div style='max_height:400px; overflow-y:auto;'>{html_main}</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='section-header'>Subpoblaciones (Turnos)</div>", unsafe_allow_html=True)
            
            def render_unified_turn_table(turno_key, title):
                st.subheader(f"► {title}")
                if turno_key in solver.subpopulations:
                    html_sub = """<table class="dark-table">
                    <thead>
                    <tr><th>Carga</th><th>Curso</th><th>Grupo</th><th>Docente</th><th>Cromosoma (Bits)</th><th>P.Sel</th></tr>
                    </thead><tbody>"""
                    has_data = False
                    for horas, idxs in solver.subpopulations[turno_key].items():
                        for idx in idxs[:10]:
                            has_data = True
                            stat = next((s for s in gene_stats if s['idx'] == idx), None)
                            if not stat: continue
                            g = best.genes[idx]
                            off = offers[idx]
                            doc_code = get_docente_code(bundle, g.teacher_id)
                            bits_doc = to_binary_string(g.teacher_id, 6)
                            bits_dias = day_to_bitmask(g.days)
                            bits_1 = f"{to_binary_string(g.room1, 4)}{to_binary_string(g.start1, 5)}"
                            if len(g.days) > 1 and g.room2 is not None:
                                bits_2 = f" {to_binary_string(g.room2, 4)}{to_binary_string(g.start2, 5)}"
                            else:
                                bits_2 = " 000000000"
                            full_bits = f"{bits_doc} {bits_dias} {bits_1}{bits_2}"
                            p_sel = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0.0
                            html_sub += f"""<tr>
                            <td>{horas} Hrs</td>
                            <td>{off.cod_curso}</td>
                            <td>{off.grupo_horario}</td>
                            <td>{doc_code}</td>
                            <td>{full_bits}</td>
                            <td>{p_sel:.6f}</td>
                            </tr>"""
                    html_sub += "</tbody></table>"
                    if has_data:
                        st.markdown(html_sub, unsafe_allow_html=True)
                    else:
                        st.info("Sin datos.")
                else:
                    st.info("No hay datos para este turno.")

            render_unified_turn_table("M", "Turno Mañana")
            render_unified_turn_table("T", "Turno Tarde")
            render_unified_turn_table("N", "Turno Noche")

            st.markdown("<div class='section-header'>Reporte de Penalidades</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            def make_list_html(items, empty_msg):
                if not items: return f"<div class='dark-info-box' style='color:#88dd88;'>{empty_msg}</div>"
                html = "<div class='dark-info-box'>"
                for item in items[:20]:
                    html += f"{item}<br>"
                html += "</div>"
                return html

            with c1:
                st.markdown("**Penalidad Docente**")
                st.markdown(make_list_html(pen_doc, "0 Conflictos"), unsafe_allow_html=True)
            with c2:
                st.markdown("**Penalidad Docente Ciclo**")
                st.markdown(make_list_html(pen_ciclo, "0 Conflictos"), unsafe_allow_html=True)
            with c3:
                st.markdown("**Penalidad Docente Aula**")
                st.markdown(make_list_html(pen_aula, "0 Conflictos"), unsafe_allow_html=True)

    # 3. HORARIO POR CICLO
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

    # 4. HORARIO POR AULA
    elif page == "Horario por Aula":
        st.header("🏫 Horario por Aula/Ambiente")
        if not st.session_state.get("has_run", False):
            st.warning("Ejecute la asignación primero.")
        else:
            offers = st.session_state.offers
            best = st.session_state.best_ind
            aula_ids = sorted(bundle.aulas["room_id"].unique())
            aula_options = {rid: get_room_code(bundle, rid) for rid in aula_ids}
            sel_aula = st.selectbox("Seleccione Aula:", options=aula_ids, format_func=lambda x: aula_options[x])
            if sel_aula:
                df_a = create_schedule_matrix(best, offers, bundle, "Aula", sel_aula, dias_map, horas_map)
                st.markdown(df_a.to_html(escape=False, classes="schedule-table"), unsafe_allow_html=True)

    # 5. HORARIOS EN GENERAL
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