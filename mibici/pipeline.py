"""
Pipeline de preparación de datos del proyecto MiBici.

Permite cargar los datos crudos y ejecutar las etapas de
preparación de manera secuencial.
"""

from mibici.dataset import load_raw
from mibici.data_preparation import quality_fixes, structure


# ============================================================
# ETAPAS DEL PIPELINE
# ============================================================

STAGES = [
    ("structure", structure.run),
    ("quality_fixes", quality_fixes.run),
]


def load_stage(stage="raw", nrows=None):
    """
    Carga los datos del proyecto y ejecuta las transformaciones
    necesarias hasta la etapa solicitada.

    Parameters
    ----------
    stage : str
        Etapa que se desea obtener.
        Opciones:
        - raw
        - structure
        - quality_fixes

    nrows : int, optional
        Número máximo de filas a cargar.
        Si es None, se cargan todos los datos.

    Returns
    -------
    dict
        Diccionario con los DataFrames procesados.
    """

    # Etapas disponibles
    nombres = ["raw"] + [nombre for nombre, _ in STAGES]

    # Validar etapa solicitada
    if stage not in nombres:
        raise ValueError(
            f"Stage desconocido: {stage}. "
            f"Opciones disponibles: {nombres}"
        )

    # ========================================================
    # 1. CARGAR DATOS CRUDOS
    # ========================================================

    datos = load_raw(nrows=nrows)

    # Si únicamente se solicitan datos crudos
    if stage == "raw":
        return datos

    # ========================================================
    # 2. EJECUTAR ETAPAS DEL PIPELINE
    # ========================================================

    for nombre, transformar in STAGES:

        datos = transformar(datos)

        # Detener el pipeline en la etapa solicitada
        if nombre == stage:
            break

    return datos
