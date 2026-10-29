"""Metriche di rendimento relativo al benchmark su dati mensili."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rendimento_annuo(r: pd.Series) -> float:
    """Rendimento composto annuo da rendimenti mensili."""
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def volatilita(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(12))


def massima_perdita(r: pd.Series) -> float:
    valore = (1 + r).cumprod()
    return float((valore / np.maximum(valore.cummax(), 1.0) - 1).min())


def tracking_error(r: pd.Series, benchmark: pd.Series) -> float:
    """Deviazione standard annua della differenza di rendimento rispetto al benchmark."""
    return float((r - benchmark).std() * np.sqrt(12))


def information_ratio(r: pd.Series, benchmark: pd.Series) -> float:
    """Rendimento attivo medio annuo diviso tracking error."""
    attivo = r - benchmark
    te = attivo.std() * np.sqrt(12)
    return float(attivo.mean() * 12 / te) if te > 1e-12 else np.nan


def beta(r: pd.Series, benchmark: pd.Series) -> float:
    return float(np.cov(r, benchmark)[0, 1] / np.var(benchmark, ddof=1))


def capture_ratio(r: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Up e down capture: quanto del rialzo e del ribasso del benchmark il fondo cattura (1 = tutto).

    Calcolati come rapporto tra i rendimenti composti annui nei mesi di rialzo e in quelli di ribasso
    del benchmark. Un buon gestore difensivo ha down capture minore di up capture.
    """
    def composto(x: pd.Series) -> float:
        return float((1 + x).prod() ** (12 / len(x)) - 1) if len(x) else np.nan

    su, giu = benchmark > 0, benchmark < 0
    return composto(r[su]) / composto(benchmark[su]), composto(r[giu]) / composto(benchmark[giu])


def sintesi(r: pd.Series, benchmark: pd.Series, risk_free: pd.Series | float = 0.0) -> dict[str, float]:
    """Indicatori principali di un fondo rispetto al benchmark (rendimenti mensili allineati)."""
    r, benchmark = r.align(benchmark, join="inner")
    rf = risk_free.reindex(r.index).fillna(0.0) if isinstance(risk_free, pd.Series) else risk_free
    eccesso = r - rf
    su, giu = capture_ratio(r, benchmark)
    return {
        "Rendimento annuo": rendimento_annuo(r),
        "Volatilità": volatilita(r),
        "Sharpe": float(eccesso.mean() / eccesso.std() * np.sqrt(12)),
        "Massima perdita": massima_perdita(r),
        "Rendimento attivo": rendimento_annuo(r) - rendimento_annuo(benchmark),
        "Tracking error": tracking_error(r, benchmark),
        "Information ratio": information_ratio(r, benchmark),
        "Beta": beta(r, benchmark),
        "Up capture": su,
        "Down capture": giu,
        "Mesi sopra il benchmark": float((r > benchmark).mean()),
    }


def tabella(fondi: pd.DataFrame, benchmark: pd.Series, risk_free: pd.Series | float = 0.0) -> pd.DataFrame:
    """Metriche di :func:`sintesi` per ogni colonna di ``fondi``."""
    return pd.DataFrame({c: sintesi(fondi[c].dropna(), benchmark, risk_free) for c in fondi}).T


def rendimenti_annuali(serie: pd.DataFrame) -> pd.DataFrame:
    """Rendimento di ogni anno solare per ogni colonna."""
    return (1 + serie).groupby(serie.index.year).prod() - 1
