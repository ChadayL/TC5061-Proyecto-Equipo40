def normalizar_columnas(df):
    resultado = df.copy()
    resultado.columns = resultado.columns.str.strip()
    return resultado


def run(datos):
    """Ejecuta las transformaciones de estructura."""
    return {
        "viajes": normalizar_columnas(datos["viajes"]),
        "estaciones": normalizar_columnas(datos["estaciones"]),
    }