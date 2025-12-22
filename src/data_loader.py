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
    # Se eliminó pref_docente_curso
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
    # Se eliminó la lectura de preferencia_docente_curso.csv
    prog = pd.read_csv(f"{data_dir}/programacion_2012B.csv")
    dias = pd.read_csv(f"{data_dir}/dias.csv")
    horas = pd.read_csv(f"{data_dir}/horas.csv")

    # Normalizaciones útiles
    # Extraer ID numérico de docentes (D01 -> 1)
    docentes["teacher_id"] = docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    
    # Extraer ID numérico de aulas (A1 -> 1, L1 -> 1, etc.)
    # Nota: Asegúrate de que esta lógica sea compatible con cómo defines los IDs en domains.py
    aulas["room_id"] = aulas["cod_aula"].astype(str).str.replace("A", "", regex=False).str.replace("L", "", regex=False).str.replace("E", "", regex=False)
    aulas["room_id"] = pd.to_numeric(aulas["room_id"], errors="coerce").fillna(0).astype(int)

    # Se eliminó la normalización de preferencias

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