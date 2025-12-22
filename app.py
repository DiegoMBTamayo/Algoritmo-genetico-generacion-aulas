# app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.config import GAConfig, TURN_SLOTS
from src.domains import build_offer_domains
from src.initial_population import build_initial_population
from src.ga import GeneticSolver
# Importamos funciones auxiliares de tu run.py
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
    
    /* ESTILOS PARA VISTA TIPO DOCUMENTACIÓN (LEGACY) */
    .doc-container {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        background-color: #f4f4f4;
        padding: 10px;
        border: 1px solid #ddd;
    }
    
    .yellow-table {
        width: 100%;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        background-color: #ffffcc; /* Amarillo Documentación */
        border: 1px solid #999;
        border-collapse: collapse;
        margin-bottom: 15px;
        color: #000;
    }
    .yellow-table th {
        background-color: #e0e0e0;
        border: 1px solid #666;
        padding: 4px;
        text-align: left;
        font-weight: bold;
    }
    .yellow-table td {
        border: 1px solid #ccc;
        padding: 2px 5px;
    }
    
    .info-box {
        background-color: #ffffcc;
        border: 1px solid #999;
        padding: 8px;
        font-family: 'Courier New', monospace;
        font-size: 11px;
        margin-bottom: 10px;
        color: #000;
    }
    
    .section-header {
        font-family: Arial, sans-serif;
        font-weight: bold;
        color: #333;
        margin-top: 15px;
        margin-bottom: 5px;
        border-bottom: 2px solid #ff4b4b;
    }

    /* Estilos para las tablas de horario gráficas */
    .schedule-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 12px;
    }
    .schedule-table th {
        background-color: #f2f2f2;
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
        font-weight: bold;
    }
    .schedule-table td {
        border: 1px solid #ddd;
        padding: 4px;
        vertical-align: top;
        background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAPEOS AUXILIARES ---
DIAS_MAP = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado"}
HORAS_MAP = {
    0: "08:00-08:50", 1: "08:50-09:40", 2: "09:40-10:30", 3: "10:30-11:20", 4: "11:20-12:10",
    5: "12:10-13:00", 6: "13:00-13:50", 7: "13:50-14:40", 8: "14:40-15:30", 9: "15:30-16:20",
    10: "16:20-17:10", 11: "17:10-18:00", 12: "18:00-18:50", 13: "18:50-19:40", 14: "19:40-20:30"
}

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

def create_schedule_matrix(best_ind, offers, bundle, filter_mode, filter_val):
    matrix = np.full((15, 6), "", dtype=object)
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
                        if 0 <= h < 15:
                            matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room1_code, off.tipo_hora)
        
        # Bloque 2
        if len(g.days) > 1 and g.room2 is not None:
            if filter_mode == "Ciclo" or g.room2 == filter_val:
                room2_code = get_room_code(bundle, g.room2)
                for d in [g.days[1]]:
                    if 0 <= d < 6:
                        for h in range(g.start2, g.start2 + g.len2):
                            if 0 <= h < 15:
                                matrix[h, d] += get_html_card(off.nombre, off.grupo_horario, doc_code, room2_code, off.tipo_hora)
    
    df_mat = pd.DataFrame(matrix, columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"])
    df_mat.index = [HORAS_MAP.get(i, str(i)) for i in range(15)]
    return df_mat

# --- HELPER: CÁLCULO DINÁMICO DE PENALIDADES ---
def calculate_dynamic_details(individual, offers, bundle):
    cfg = GAConfig()
    n_cycles = 10
    n_rooms = int(bundle.aulas["room_id"].max()) + 1
    if "tid_num" not in bundle.docentes.columns:
        bundle.docentes["tid_num"] = bundle.docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    n_teachers = int(bundle.docentes["tid_num"].max()) + 1

    count_cycle = np.zeros((n_cycles + 1, 6, 15), dtype=int)
    count_room = np.zeros((n_rooms + 1, 6, 15), dtype=int)
    count_teach = np.zeros((n_teachers + 1, 6, 15), dtype=int)

    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        cyc = int(off.ciclo) - 1
        for h in range(g.start1, g.start1 + g.len1):
            if h < 15 and g.days[0] < 6:
                count_cycle[cyc, g.days[0], h] += 1
                count_room[g.room1, g.days[0], h] += 1
                count_teach[g.teacher_id, g.days[0], h] += 1
        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < 15 and g.days[1] < 6:
                    count_cycle[cyc, g.days[1], h] += 1
                    count_room[g.room2, g.days[1], h] += 1
                    count_teach[g.teacher_id, g.days[1], h] += 1

    gene_stats = []
    penalties_docente = {}
    total_fitness_sum = 0
    
    for i, (g, off) in enumerate(zip(individual.genes, offers)):
        pen_gen = 0
        cyc = int(off.ciclo) - 1
        
        for h in range(g.start1, g.start1 + g.len1):
            if h < 15 and g.days[0] < 6:
                if count_cycle[cyc, g.days[0], h] > 1: pen_gen += cfg.W_CONFLICT_CYCLE
                if count_room[g.room1, g.days[0], h] > 1: pen_gen += cfg.W_CONFLICT_ROOM
                if count_teach[g.teacher_id, g.days[0], h] > 1: pen_gen += cfg.W_CONFLICT_TEACHER

        if len(g.days) > 1 and g.room2 is not None:
            for h in range(g.start2, g.start2 + g.len2):
                if h < 15 and g.days[1] < 6:
                    if count_cycle[cyc, g.days[1], h] > 1: pen_gen += cfg.W_CONFLICT_CYCLE
                    if count_room[g.room2, g.days[1], h] > 1: pen_gen += cfg.W_CONFLICT_ROOM
                    if count_teach[g.teacher_id, g.days[1], h] > 1: pen_gen += cfg.W_CONFLICT_TEACHER
        
        aptitud = 1.0 / (1.0 + pen_gen)
        total_fitness_sum += aptitud
        gene_stats.append({
            "idx": i,
            "penal": pen_gen,
            "aptitud": aptitud
        })
        
        if pen_gen > 0:
            if g.teacher_id not in penalties_docente: penalties_docente[g.teacher_id] = 0
            penalties_docente[g.teacher_id] += pen_gen

    return gene_stats, total_fitness_sum, penalties_docente

# --- MAIN APP ---
def main():
    if 'bundle' not in st.session_state:
        bundle = load_data("data")
        st.session_state.bundle = bundle
    bundle = st.session_state.bundle

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

    # 1. REALIZAR ASIGNACIÓN
    if page == "Realizar Asignación":
        st.header("📋 Realizar Asignación")
        
        tab_listas = st.tabs(["Docentes", "Aulas", "Cursos", "Clase Fija"])
        
        # Filtramos columnas internas para la vista
        # DOCENTES: Ocultamos teacher_id y tid_num
        cols_doc = [c for c in bundle.docentes.columns if c not in ['teacher_id', 'tid_num']]
        with tab_listas[0]: st.dataframe(bundle.docentes[cols_doc], height=150)
        
        # AULAS: Ocultamos room_id
        cols_aulas = [c for c in bundle.aulas.columns if c not in ['room_id', 'es_laboratorio']]
        with tab_listas[1]: st.dataframe(bundle.aulas[cols_aulas], height=150)
        
        with tab_listas[2]: st.dataframe(bundle.cursos, height=150)
        with tab_listas[3]: st.dataframe(bundle.clase_fija, height=150)
        
        st.divider()

        # Configuración Oculta (Valores por defecto según documentación)
        pop_size = 102
        generations = 100
        mutation_rate = 0.1

        # BOTÓN ÚNICO
        if st.button("🚀 CREAR POBLACIÓN INICIAL"):
            with st.status("Ejecutando proceso...", expanded=True) as status:
                st.write("Generando ofertas...")
                offers = build_offers(bundle)
                baseline = build_baseline_lab(bundle)
                aulas_calc = bundle.aulas.copy()
                aulas_calc["es_laboratorio"] = aulas_calc["cod_tipo"].astype(str).str.startswith("L")
                
                domains = build_offer_domains(offers, aulas_calc, bundle.docentes, TURN_SLOTS, 6, baseline)
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

        # MUESTRA DE POBLACIÓN INICIAL
        if st.session_state.get("has_run", False):
            st.divider()
            st.subheader("Población Inicial (Aleatoria)")
            st.caption("A continuación se muestran los cromosomas generados en la población inicial.")
            
            sample_ind = st.session_state.initial_pop_list[0]
            offers = st.session_state.offers
            
            code_lines = []
            header = f"{'Curso':<10} {'Grupo':<8} {'Doc(6b)':<8} {'Días(6b)':<8} {'Aula1(4b)':<10} {'Hora1(5b)':<10} {'Aula2(4b)':<10} {'Hora2(5b)':<10}"
            code_lines.append(header)
            code_lines.append("-" * len(header))
            
            for g, off in zip(sample_ind.genes, offers):
                etiqueta = f"{off.cod_curso}"
                grp = f"{off.grupo_horario}"
                doc_bin = to_binary_string(g.teacher_id, 6)
                dias_bin = day_to_bitmask(g.days)
                a1 = to_binary_string(g.room1, 4)
                h1 = to_binary_string(g.start1, 5)
                a2 = to_binary_string(g.room2, 4)
                h2 = to_binary_string(g.start2, 5)
                code_lines.append(f"{etiqueta:<10} {grp:<8} {doc_bin:<8} {dias_bin:<8} {a1:<10} {h1:<10} {a2:<10} {h2:<10}")
            
            st.text_area("Cromosomas (Individuo 0)", "\n".join(code_lines), height=300)

    # 2. PROCESOS ADICIONALES
    elif page == "Procesos Adicionales":
        st.header("⚙️ Procesos Adicionales")
        
        if not st.session_state.get("has_run", False):
            st.warning("Ejecute la asignación primero.")
        else:
            solver = st.session_state.solver
            best = st.session_state.best_ind
            offers = st.session_state.offers

            gene_stats, total_fit_sum, pen_docente = calculate_dynamic_details(best, offers, bundle)

            st.markdown("<div class='section-header'>Población y Probabilidades de Selección (Detalle por Curso)</div>", unsafe_allow_html=True)
            
            html = """
            <table class="yellow-table">
                <thead>
                    <tr>
                        <th>Curso</th>
                        <th>Grupo</th>
                        <th>Docente</th>
                        <th>Cromosoma (Docente|Días|Aula|Hora)</th>
                        <th>Penal</th>
                        <th>Aptitud</th>
                        <th>P.Selec</th>
                        <th>P.Acumul</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            p_acum = 0.0
            for stat in gene_stats[:20]:
                idx = stat['idx']
                g = best.genes[idx]
                off = offers[idx]
                doc_code = get_docente_code(bundle, g.teacher_id)
                bits_doc = to_binary_string(g.teacher_id, 6)
                bits_dias = day_to_bitmask(g.days)
                bits_bloque1 = f"{to_binary_string(g.room1, 4)}{to_binary_string(g.start1, 5)}"
                bits_bloque2 = f"{to_binary_string(g.room2, 4)}{to_binary_string(g.start2, 5)}"
                full_bits = f"{bits_doc} {bits_dias} {bits_bloque1} {bits_bloque2}"
                p_selec = stat['aptitud'] / total_fit_sum if total_fit_sum > 0 else 0
                p_acum += p_selec
                
                html += f"""
                    <tr>
                        <td>{off.cod_curso}</td>
                        <td>{off.grupo_horario}</td>
                        <td>{doc_code}</td>
                        <td>{full_bits}</td>
                        <td>{stat['penal']}</td>
                        <td>{stat['aptitud']:.6f}</td>
                        <td>{p_selec:.6f}</td>
                        <td>{p_acum:.6f}</td>
                    </tr>
                """
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
            
            st.markdown("<div class='section-header'>Subpoblaciones (Turnos)</div>", unsafe_allow_html=True)
            col_m, col_t, col_n = st.columns(3)
            
            def render_doc_subpop(turno_key, title, col):
                with col:
                    st.markdown(f"**{title}**")
                    if turno_key in solver.subpopulations:
                        for horas, idxs in solver.subpopulations[turno_key].items():
                            box_html = f"<div class='info-box'><strong>[Teoria] ({horas} Horas)</strong><br><br>"
                            for idx in idxs[:6]:
                                off = offers[idx]
                                g = best.genes[idx]
                                doc = get_docente_code(bundle, g.teacher_id)
                                bits = f"{to_binary_string(g.teacher_id, 6)} {day_to_bitmask(g.days)}..."
                                box_html += f"{off.cod_curso} {off.grupo_horario} {doc} {bits}<br>"
                            if len(idxs) > 6: box_html += "..."
                            box_html += "</div>"
                            st.markdown(box_html, unsafe_allow_html=True)
            
            render_doc_subpop("M", "Turno Mañana", col_m)
            render_doc_subpop("T", "Turno Tarde", col_t)
            render_doc_subpop("N", "Turno Noche", col_n)

            st.markdown("<div class='section-header'>Reporte de Penalidades</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Penalidad por Docente**")
                pen_html = "<div class='info-box'>"
                if not pen_docente:
                    pen_html += "No se encontraron conflictos de docentes."
                else:
                    for tid, pval in list(pen_docente.items())[:10]:
                        dname = get_docente_name(bundle, tid)
                        dcode = get_docente_code(bundle, tid)
                        pen_html += f"{dcode} {dname} : {pval}<br>"
                pen_html += "</div>"
                st.markdown(pen_html, unsafe_allow_html=True)
            with c2:
                st.markdown("**Resumen Global**")
                fit_html = f"""
                <div class='info-box' style='font-size:14px;'>
                    Población: {len(st.session_state.initial_pop_list) if 'initial_pop_list' in st.session_state else 'N/A'}<br>
                    <hr>
                    <strong>Fitness Final: {best.fitness:.6f}</strong><br>
                    <strong>Penalidad Total: {int(best.penalty)}</strong>
                </div>
                """
                st.markdown(fit_html, unsafe_allow_html=True)

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
                df_c = create_schedule_matrix(best, offers, bundle, "Ciclo", sel_ciclo)
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
                df_a = create_schedule_matrix(best, offers, bundle, "Aula", sel_aula)
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
                        "Día": DIAS_MAP.get(d_idx, "-"),
                        "Hora": HORAS_MAP.get(start, "-"),
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