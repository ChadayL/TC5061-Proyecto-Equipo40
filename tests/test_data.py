from mibici.dataset import load_raw
from mibici.pipeline import load_stage
from mibici.features import (
    build_p1_features,
    build_p2_features,
    build_p3_features,
)


def test_load_raw():
    """Verifica que los datos principales se carguen correctamente."""

    datos = load_raw()

    assert "viajes" in datos
    assert "estaciones" in datos
    assert len(datos["viajes"]) > 0
    assert len(datos["estaciones"]) > 0


def test_pipeline_raw():
    """Verifica la etapa raw del pipeline."""

    datos = load_stage("raw")

    assert "viajes" in datos
    assert "estaciones" in datos


def test_pipeline_structure():
    """Verifica la etapa structure."""

    datos = load_stage("structure")

    assert "viajes" in datos
    assert "estaciones" in datos


def test_pipeline_quality_fixes():
    """Verifica la etapa quality_fixes."""

    datos = load_stage("quality_fixes")

    assert "viajes" in datos
    assert "estaciones" in datos


def test_p1_features():
    """Verifica la construcción de variables para P1."""

    datos = load_raw()

    p1 = build_p1_features(
        datos["viajes"],
        datos["estaciones"],
    )

    columnas_esperadas = [
        "fecha",
        "Origen_Id",
        "viajes",
        "dia_semana",
        "fin_semana",
        "lag_1",
        "lag_7",
        "promedio_7d",
        "latitude",
        "longitude",
        "capacity",
        "viajes_manana",
    ]

    for columna in columnas_esperadas:
        assert columna in p1.columns

    assert len(p1) > 0


def test_p2_features():
    """Verifica la construcción de variables para P2."""

    datos = load_raw()

    p2 = build_p2_features(
        datos["viajes"],
        datos["estaciones"],
    )

    columnas_esperadas = [
        "fecha",
        "estacion_id",
        "salidas",
        "llegadas",
        "balance",
        "capacity",
        "presion_capacidad",
        "balance_lag_1",
        "balance_lag_7",
        "presion_lag_1",
        "presion_promedio_7d",
        "riesgo_actual",
        "riesgo_manana",
    ]

    for columna in columnas_esperadas:
        assert columna in p2.columns

    assert len(p2) > 0


def test_p2_target_binary():
    """Verifica que el target de P2 sea binario."""

    datos = load_raw()

    p2 = build_p2_features(
        datos["viajes"],
        datos["estaciones"],
    )

    valores = set(
        p2["riesgo_manana"].unique()
    )

    assert valores.issubset({0, 1})


def test_p3_features():
    """Verifica la construcción de variables para P3."""

    datos = load_raw()

    p3 = build_p3_features(
        datos["viajes"],
        datos["estaciones"],
    )

    columnas_esperadas = [
        "estacion_id",
        "salidas_totales",
        "llegadas_totales",
        "salidas_promedio",
        "salidas_std",
        "proporcion_fin_semana",
        "viajes_totales",
        "balance",
        "ratio_llegadas_salidas",
        "latitude",
        "longitude",
        "capacity",
    ]

    for columna in columnas_esperadas:
        assert columna in p3.columns

    assert len(p3) > 0

    # Debe existir una sola fila por estación.
    assert p3["estacion_id"].is_unique

    # Las estaciones utilizadas deben tener capacidad válida.
    assert (p3["capacity"] > 0).all()