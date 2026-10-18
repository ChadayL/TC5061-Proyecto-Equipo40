from sklearn.preprocessing import OneHotEncoder


def build_encoder():
    """Crea un encoder sin ajustar."""
    return OneHotEncoder(handle_unknown="ignore")