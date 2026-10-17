import pandas as pd


# ============================================================
# P1 - REGRESIÓN
# ============================================================

def build_p1_features(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para P1.

    P1:
    ¿Cuántos viajes se iniciarán en una estación
    en un periodo futuro?

    Definición:
    - Tipo de problema: Regresión
    - Granularidad: estación x día
    - Variable objetivo: número de viajes iniciados
    - Horizonte: día siguiente

    Las variables históricas utilizan únicamente información
    anterior al día que se quiere predecir para evitar
    data leakage.
    """

    # ========================================================
    # 1. COPIAR LOS DATOS
    # ========================================================

    df = viajes.copy()
    estaciones = estaciones.copy()

    # ========================================================
    # 2. PREPARAR FECHA
    # ========================================================

    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"],
        errors="coerce",
    )

    # Eliminar registros sin fecha válida
    df = df.dropna(subset=["Inicio_del_viaje"])

    # Obtener solamente la fecha
    df["fecha"] = df["Inicio_del_viaje"].dt.normalize()

    # ========================================================
    # 3. CONTAR VIAJES POR ESTACIÓN Y DÍA
    # ========================================================

    conteos = (
        df.groupby(["fecha", "Origen_Id"])
        .size()
        .reset_index(name="viajes")
    )

    # ========================================================
    # 4. INCLUIR DÍAS CON CERO VIAJES
    # ========================================================

    fechas = pd.date_range(
        start=df["fecha"].min(),
        end=df["fecha"].max(),
        freq="D",
    )

    ids_estaciones = (
        estaciones["id"]
        .dropna()
        .unique()
    )

    indice_completo = pd.MultiIndex.from_product(
        [fechas, ids_estaciones],
        names=["fecha", "Origen_Id"],
    )

    diario = (
        conteos
        .set_index(["fecha", "Origen_Id"])
        .reindex(indice_completo, fill_value=0)
        .reset_index()
    )

    # ========================================================
    # 5. ORDENAR CRONOLÓGICAMENTE
    # ========================================================

    diario = diario.sort_values(
        ["Origen_Id", "fecha"]
    )

    # ========================================================
    # 6. VARIABLES DE CALENDARIO
    # ========================================================

    # Lunes = 0
    # Domingo = 6
    diario["dia_semana"] = (
        diario["fecha"].dt.dayofweek
    )

    diario["fin_semana"] = (
        diario["dia_semana"] >= 5
    ).astype(int)

    # ========================================================
    # 7. VARIABLES HISTÓRICAS
    # ========================================================

    # Viajes del día anterior
    diario["lag_1"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .shift(1)
    )

    # Viajes de hace 7 días
    diario["lag_7"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .shift(7)
    )

    # Promedio de los 7 días anteriores
    diario["promedio_7d"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    # ========================================================
    # 8. INFORMACIÓN DE LAS ESTACIONES
    # ========================================================

    estaciones_p1 = estaciones[
        [
            "id",
            "latitude",
            "longitude",
            "dpcapacity",
        ]
    ].copy()

    # Renombrar capacidad
    estaciones_p1 = estaciones_p1.rename(
        columns={"dpcapacity": "capacity"}
    )

    # Incorporar información de estación
    diario = diario.merge(
        estaciones_p1,
        left_on="Origen_Id",
        right_on="id",
        how="left",
    )

    # Eliminar id duplicado
    diario = diario.drop(columns=["id"])

    # Convertir capacidad a número
    diario["capacity"] = pd.to_numeric(
        diario["capacity"],
        errors="coerce",
    )

    # ========================================================
    # 9. CREAR TARGET DEL DÍA SIGUIENTE
    # ========================================================

    diario["viajes_manana"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .shift(-1)
    )

    # ========================================================
    # 10. ELIMINAR FILAS SIN INFORMACIÓN SUFICIENTE
    # ========================================================

    diario = diario.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "promedio_7d",
            "viajes_manana",
            "capacity",
        ]
    )

    # ========================================================
    # 11. REINICIAR ÍNDICE
    # ========================================================

    diario = diario.reset_index(drop=True)

    return diario


# ============================================================
# P2 - CLASIFICACIÓN
# ============================================================

def build_p2_features(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para P2.

    P2:
    ¿Una estación estará en riesgo de quedarse sin bicicletas
    o sin anclajes disponibles en un periodo futuro?

    Definición:
    - Tipo de problema: Clasificación binaria
    - Granularidad: estación x día
    - Horizonte: día siguiente
    - Target:
        1 = estación en riesgo mañana
        0 = estación sin riesgo mañana

    Como MiBici no proporciona inventario histórico de bicicletas,
    el riesgo se aproxima utilizando el desbalance entre salidas
    y llegadas en relación con la capacidad de la estación.
    """

    # ========================================================
    # 1. COPIAR LOS DATOS
    # ========================================================

    df = viajes.copy()
    estaciones = estaciones.copy()

    # ========================================================
    # 2. PREPARAR FECHAS
    # ========================================================

    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"],
        errors="coerce",
    )

    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"],
        errors="coerce",
    )

    # Eliminar fechas inválidas
    df = df.dropna(
        subset=[
            "Inicio_del_viaje",
            "Fin_del_viaje",
        ]
    )

    df["fecha_salida"] = (
        df["Inicio_del_viaje"].dt.normalize()
    )

    df["fecha_llegada"] = (
        df["Fin_del_viaje"].dt.normalize()
    )

    # ========================================================
    # 3. CONTAR SALIDAS POR ESTACIÓN Y DÍA
    # ========================================================

    salidas = (
        df.groupby(
            ["fecha_salida", "Origen_Id"]
        )
        .size()
        .reset_index(name="salidas")
        .rename(
            columns={
                "fecha_salida": "fecha",
                "Origen_Id": "estacion_id",
            }
        )
    )

    # ========================================================
    # 4. CONTAR LLEGADAS POR ESTACIÓN Y DÍA
    # ========================================================

    llegadas = (
        df.groupby(
            ["fecha_llegada", "Destino_Id"]
        )
        .size()
        .reset_index(name="llegadas")
        .rename(
            columns={
                "fecha_llegada": "fecha",
                "Destino_Id": "estacion_id",
            }
        )
    )

    # ========================================================
    # 5. CREAR TODAS LAS COMBINACIONES ESTACIÓN × DÍA
    # ========================================================

    fecha_min = min(
        df["fecha_salida"].min(),
        df["fecha_llegada"].min(),
    )

    fecha_max = max(
        df["fecha_salida"].max(),
        df["fecha_llegada"].max(),
    )

    fechas = pd.date_range(
        start=fecha_min,
        end=fecha_max,
        freq="D",
    )

    ids_estaciones = (
        estaciones["id"]
        .dropna()
        .unique()
    )

    indice_completo = pd.MultiIndex.from_product(
        [fechas, ids_estaciones],
        names=["fecha", "estacion_id"],
    )

    diario = (
        pd.DataFrame(index=indice_completo)
        .reset_index()
    )

    # ========================================================
    # 6. INCORPORAR SALIDAS Y LLEGADAS
    # ========================================================

    diario = diario.merge(
        salidas,
        on=["fecha", "estacion_id"],
        how="left",
    )

    diario = diario.merge(
        llegadas,
        on=["fecha", "estacion_id"],
        how="left",
    )

    diario["salidas"] = (
        diario["salidas"]
        .fillna(0)
        .astype(int)
    )

    diario["llegadas"] = (
        diario["llegadas"]
        .fillna(0)
        .astype(int)
    )

    # ========================================================
    # 7. AGREGAR INFORMACIÓN DE LA ESTACIÓN
    # ========================================================

    estaciones_p2 = estaciones[
        [
            "id",
            "latitude",
            "longitude",
            "dpcapacity",
        ]
    ].copy()

    estaciones_p2 = estaciones_p2.rename(
        columns={"dpcapacity": "capacity"}
    )

    diario = diario.merge(
        estaciones_p2,
        left_on="estacion_id",
        right_on="id",
        how="left",
    )

    diario = diario.drop(
        columns=["id"]
    )

    # Capacidad debe ser numérica
    diario["capacity"] = pd.to_numeric(
        diario["capacity"],
        errors="coerce",
    )

    # Eliminar estaciones sin capacidad
    diario = diario.dropna(
        subset=["capacity"]
    )

    # Evitar capacidad cero
    diario = diario[
        diario["capacity"] > 0
    ].copy()

    # ========================================================
    # 8. ORDENAR CRONOLÓGICAMENTE
    # ========================================================

    diario = diario.sort_values(
        ["estacion_id", "fecha"]
    )

    # ========================================================
    # 9. VARIABLES DE CALENDARIO
    # ========================================================

    diario["dia_semana"] = (
        diario["fecha"].dt.dayofweek
    )

    diario["fin_semana"] = (
        diario["dia_semana"] >= 5
    ).astype(int)

    # ========================================================
    # 10. BALANCE DE VIAJES
    # ========================================================

    # Positivo:
    # más bicicletas salieron que llegaron.
    #
    # Negativo:
    # más bicicletas llegaron que salieron.

    diario["balance"] = (
        diario["salidas"]
        - diario["llegadas"]
    )

    # Presión relativa con respecto
    # a la capacidad de la estación

    diario["presion_capacidad"] = (
        diario["balance"].abs()
        / diario["capacity"]
    )

    # ========================================================
    # 11. VARIABLES HISTÓRICAS
    # ========================================================

    # Balance del día anterior
    diario["balance_lag_1"] = (
        diario
        .groupby("estacion_id")["balance"]
        .shift(1)
    )

    # Balance de hace 7 días
    diario["balance_lag_7"] = (
        diario
        .groupby("estacion_id")["balance"]
        .shift(7)
    )

    # Presión del día anterior
    diario["presion_lag_1"] = (
        diario
        .groupby("estacion_id")["presion_capacidad"]
        .shift(1)
    )

    # Promedio de presión de los
    # siete días anteriores
    diario["presion_promedio_7d"] = (
        diario
        .groupby("estacion_id")["presion_capacidad"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(7)
                .mean()
            )
        )
    )

    # ========================================================
    # 12. DEFINIR RIESGO ACTUAL
    # ========================================================

    # Una estación se considera en riesgo
    # cuando el desbalance diario equivale
    # al menos al 50 % de su capacidad.

    UMBRAL_RIESGO = 0.50

    diario["riesgo_actual"] = (
        diario["presion_capacidad"]
        >= UMBRAL_RIESGO
    ).astype(int)

    # ========================================================
    # 13. CREAR TARGET DEL DÍA SIGUIENTE
    # ========================================================

    diario["riesgo_manana"] = (
        diario
        .groupby("estacion_id")["riesgo_actual"]
        .shift(-1)
    )

    # ========================================================
    # 14. ELIMINAR FILAS SIN HISTORIA O TARGET
    # ========================================================

    diario = diario.dropna(
        subset=[
            "balance_lag_1",
            "balance_lag_7",
            "presion_lag_1",
            "presion_promedio_7d",
            "riesgo_manana",
        ]
    )

    # Convertir target a entero
    diario["riesgo_manana"] = (
        diario["riesgo_manana"]
        .astype(int)
    )

    # ========================================================
    # 15. REINICIAR ÍNDICE
    # ========================================================

    diario = diario.reset_index(
        drop=True
    )

    return diario