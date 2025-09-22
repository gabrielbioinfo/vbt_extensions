import pandas as pd


def filter_sessions(index: pd.DatetimeIndex, *, start_hour=10, end_hour=17) -> pd.Series:
    """Retorna máscara booleana True apenas durante a janela de sessão [start_hour, end_hour]."""
    hours = index.hour
    return (hours >= start_hour) & (hours < end_hour)


def session_mask(
    index: pd.DatetimeIndex,
    start: str = "09:00",
    end: str = "18:00",
    tz: str | None = None,
    weekdays: list[int] | None = None,
) -> pd.Series:
    """
    Cria máscara booleana para manter apenas barras dentro de uma sessão de mercado.

    Parameters
    ----------
    index : DatetimeIndex
        Índice temporal da série.
    start : str
        Hora de início no formato 'HH:MM'.
    end : str
        Hora de fim no formato 'HH:MM'.
    tz : str, optional
        Timezone para converter antes do filtro (ex: 'America/Sao_Paulo').
    weekdays : list[int], optional
        Dias da semana permitidos (0=Segunda, 6=Domingo). Default=None (todos).

    Returns
    -------
    pd.Series
        Série booleana True para barras dentro da sessão.
    """
    idx = index
    if tz:
        idx = idx.tz_convert(tz)

    t = idx.time
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    start_t = pd.Timestamp(0).replace(hour=s_h, minute=s_m).time()
    end_t = pd.Timestamp(0).replace(hour=e_h, minute=e_m).time()

    mask = pd.Series([(ti >= start_t) and (ti <= end_t) for ti in t], index=index)

    if weekdays is not None:
        mask &= idx.weekday.isin(weekdays)

    return mask


def apply_session_mask(series: pd.Series, mask: pd.Series, fill: bool = False) -> pd.Series:
    """Aplica máscara de sessão a uma série: mantém valores apenas onde mask=True."""
    if fill:
        return series.where(mask).ffill()
    return series.where(mask)