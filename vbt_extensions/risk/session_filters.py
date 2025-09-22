import pandas as pd


def filter_sessions(index: pd.DatetimeIndex, *, start_hour=10, end_hour=17) -> pd.Series:
    """Retorna máscara booleana True apenas durante a janela de sessão [start_hour, end_hour]."""
    hours = index.hour
    return (hours >= start_hour) & (hours < end_hour)
