from mibici.dataset import load_raw
from mibici.data_preparation import quality_fixes, structure

# Orden de ejecución
STAGES = [
    ("structure", structure.run),
    ("quality_fixes", quality_fixes.run),
]


def load_stage(stage="raw", nrows=None):
    stage_names = ["raw"] + [name for name, _ in STAGES] + ["common"]

    if stage not in stage_names:
        raise ValueError(f"Stage desconocido: {stage}. Opciones: {stage_names}")

    data = load_raw(nrows=nrows)

    if stage == "raw":
        return data

    for name, transform in STAGES:
        data = transform(data)

        if name == stage:
            break

    return data
