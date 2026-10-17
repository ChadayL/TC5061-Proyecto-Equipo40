from mibici.pipeline import load_stage
from mibici.features import build_daily_departures


def load_p1_data():
    data = load_stage("common")
    return build_daily_departures(data["viajes"])