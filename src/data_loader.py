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
    pref_docente_curso: pd.DataFrame
    programacion: pd.DataFrame
    dias: pd.DataFrame
    horas: pd.DataFrame

def load_data(data_dir: str) -> DataBundle:
    docentes = pd.read_csv(f"{data_dir}/docentes.csv")
    disponibilidad = pd.read_csv(f"{data_dir}/disponibilidad.csv")
    aulas = pd.read_csv(f"{data_dir}/aulas.csv")
    cursos = pd.read_csv(f"{data_dir}/cursos.csv")
    clase_fija = pd.read_csv(f"{data_dir}/clase_fija.csv")
    pref = pd.read_csv(f"{data_dir}/preferencia_docente_curso.csv")
    prog = pd.read_csv(f"{data_dir}/programacion_2012B.csv")
    dias = pd.read_csv(f"{data_dir}/dias.csv")
    horas = pd.read_csv(f"{data_dir}/horas.csv")

    # Normalizaciones útiles
    docentes["teacher_id"] = docentes["cod_docente"].str.replace("D", "", regex=False).astype(int)
    aulas["room_id"] = aulas["cod_aula"].astype(str).str.replace("A", "", regex=False).str.replace("L", "", regex=False)
    aulas["room_id"] = pd.to_numeric(aulas["room_id"], errors="coerce").fillna(0).astype(int)

    pref["teacher_id"] = pref["cod_docente"].str.replace("D", "", regex=False).astype(int)

    return DataBundle(
        docentes=docentes,
        disponibilidad=disponibilidad,
        aulas=aulas,
        cursos=cursos,
        clase_fija=clase_fija,
        pref_docente_curso=pref,
        programacion=prog,
        dias=dias,
        horas=horas
    )
