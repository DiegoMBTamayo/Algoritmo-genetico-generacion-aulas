# src/data_loader.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

@dataclass(frozen=True)
class DataBundle:
    docentes: pd.DataFrame
    disponibilidad: pd.DataFrame
    aulas: pd.DataFrame
    cursos: pd.DataFrame
    clase_fija: pd.DataFrame
    programacion: pd.DataFrame
    dias: pd.DataFrame
    horas: pd.DataFrame

def load_data(data_dir: str) -> DataBundle:
    # Cargar CSVs existentes
    docentes = pd.read_csv(f"{data_dir}/docentes.csv")
    disponibilidad = pd.read_csv(f"{data_dir}/disponibilidad.csv")
    aulas = pd.read_csv(f"{data_dir}/aulas.csv")
    cursos = pd.read_csv(f"{data_dir}/cursos.csv")
    clase_fija = pd.read_csv(f"{data_dir}/clase_fija.csv")
    prog = pd.read_csv(f"{data_dir}/programacion_2012B.csv")
    dias = pd.read_csv(f"{data_dir}/dias.csv")
    horas = pd.read_csv(f"{data_dir}/horas.csv")

    # --- NORMALIZACIÓN DE IDs ---
    
    # 1. Docentes: D01 -> 1
    docentes["teacher_id"] = docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    
    # 2. Aulas: CORRECCIÓN CRÍTICA
    # Antes: A1->1, L1->1 (Causaba colisión y ocultaba laboratorios)
    # Ahora: Asignamos un ID secuencial único (0, 1, 2...) basado en el orden de la tabla
    aulas["room_id"] = range(len(aulas))

    return DataBundle(
        docentes=docentes,
        disponibilidad=disponibilidad,
        aulas=aulas,
        cursos=cursos,
        clase_fija=clase_fija,
        programacion=prog,
        dias=dias,
        horas=horas
    )