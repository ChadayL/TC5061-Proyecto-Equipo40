import pandas as pd


def build_p1_features(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el dataset de Machine Learning para P1.

    P1:
    ¿Cuántos viajes se iniciarán en una estación
    en un periodo futuro?

    Definición utilizada:
    - Tipo de problema: Regresión
    - Granularidad: estación × día
    - Variable objetivo: número de viajes iniciados
    - Horizonte: día siguiente

    Las variables históricas utilizan únicamente información
    anterior al día que se quiere predecir para evitar
    data leakage.
    """

    # ---------------------------------------------------------
    # 1. COPIAR LOS DATOS
    # ---------------------------------------------------------

    df = viajes.copy()

    # ---------------------------------------------------------
    # 2. PREPARAR LA FECHA
    # ---------------------------------------------------------

    # Convertir la columna de inicio del viaje a datetime
    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"]
    )

    # Obtener solamente la fecha del viaje
    df["fecha"] = df["Inicio_del_viaje"].dt.normalize()

    # ---------------------------------------------------------
    # 3. CONTAR VIAJES POR ESTACIÓN Y DÍA
    # ---------------------------------------------------------

    conteos = (
        df.groupby(["fecha", "Origen_Id"])
        .size()
        .reset_index(name="viajes")
    )

    # ---------------------------------------------------------
    # 4. INCLUIR DÍAS CON CERO VIAJES
    # ---------------------------------------------------------

    # Crear todas las fechas existentes entre el primer
    # y último día del periodo analizado
    fechas = pd.date_range(
        start=df["fecha"].min(),
        end=df["fecha"].max(),
        freq="D",
    )

    # Obtener todos los identificadores de estaciones
    ids_estaciones = (
        estaciones["id"]
        .dropna()
        .unique()
    )

    # Crear todas las combinaciones posibles:
    # fecha × estación
    indice_completo = pd.MultiIndex.from_product(
        [fechas, ids_estaciones],
        names=["fecha", "Origen_Id"],
    )

    # Agregar las combinaciones que no tuvieron viajes
    # y asignarles viajes = 0
    diario = (
        conteos
        .set_index(["fecha", "Origen_Id"])
        .reindex(indice_completo, fill_value=0)
        .reset_index()
    )

    # ---------------------------------------------------------
    # 5. ORDENAR CRONOLÓGICAMENTE
    # ---------------------------------------------------------

    diario = diario.sort_values(
        ["Origen_Id", "fecha"]
    )

    # ---------------------------------------------------------
    # 6. VARIABLES DE CALENDARIO
    # ---------------------------------------------------------

    # Lunes = 0
    # Martes = 1
    # ...
    # Domingo = 6
    diario["dia_semana"] = (
        diario["fecha"].dt.dayofweek
    )

    # 1 = sábado o domingo
    # 0 = lunes a viernes
    diario["fin_semana"] = (
        diario["dia_semana"] >= 5
    ).astype(int)

    # ---------------------------------------------------------
    # 7. VARIABLES HISTÓRICAS
    # ---------------------------------------------------------

    # Viajes realizados el día anterior
    diario["lag_1"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .shift(1)
    )

    # Viajes realizados 7 días antes
    diario["lag_7"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .shift(7)
    )

    # Promedio de viajes de los 7 días anteriores
    #
    # shift(1) es muy importante:
    # evita utilizar los viajes del propio día que
    # estamos intentando predecir.
    diario["promedio_7d"] = (
        diario
        .groupby("Origen_Id")["viajes"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    # ---------------------------------------------------------
    # 8. INFORMACIÓN DE LAS ESTACIONES
    # ---------------------------------------------------------

    estaciones_p1 = estaciones[
        [
            "id",
            "latitude",
            "longitude",
            "dpcapacity",
        ]
    ].copy()

    # Incorporar capacidad y ubicación de cada estación
    diario = diario.merge(
        estaciones_p1,
        left_on="Origen_Id",
        right_on="id",
        how="left",
    )

    # Ya no necesitamos la columna id duplicada
    diario = diario.drop(columns=["id"])

    # ---------------------------------------------------------
    # 9. ELIMINAR FILAS SIN HISTORIA SUFICIENTE
    # ---------------------------------------------------------

    # Los primeros días de cada estación no tienen todavía
    # información suficiente para calcular lag_7 y promedio_7d.
    diario = diario.dropna(
        subset=[
            "lag_1",
            "lag_7",
            "promedio_7d",
        ]
    )

    # Reiniciar índice
    diario = diario.reset_index(drop=True)

    # ---------------------------------------------------------
    # 10. DEVOLVER DATASET FINAL
    # ---------------------------------------------------------

    return diario
