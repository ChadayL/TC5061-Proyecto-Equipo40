from mibici.dataset import load_raw
from mibici.data_preparation import structure, missing

# Orden de ejecución
STAGES = [
    ("structure", structure.run),
    ("missing", missing.run),
]


def load_stage(stage="raw", nrows=None):
    nombres = ["raw"] + [nombre for nombre, _ in STAGES]

    if stage not in nombres:
        raise ValueError(f"Stage desconocido: {stage}. Opciones: {nombres}")

    datos = load_raw(nrows=nrows)

    if stage == "raw":
        return datos

    for nombre, transformar in STAGES:
        datos = transformar(datos)

        if nombre == stage:
            break

    return datos
