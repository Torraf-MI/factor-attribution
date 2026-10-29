"""Returns-based style analysis di Sharpe (1992): lo stile effettivo di un fondo stimato dai soli rendimenti.

Si cerca la combinazione di indici di mercato, con pesi positivi che sommano a 1, che replica meglio il fondo,
cioè che rende minima la varianza della differenza di rendimento. Non servono le posizioni del fondo: bastano
i valori quota pubblicati.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class Stile:
    """Pesi di stile, qualità della replica e rendimento di selezione (annui)."""

    pesi: pd.Series
    r2: float  # quota della varianza del fondo spiegata dallo stile
    selezione: float  # rendimento medio annuo del fondo meno quello del portafoglio di stile
    tracking_error: float  # deviazione standard annua della differenza
    differenze: pd.Series = field(repr=False)


def _pesi_stile(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    k = x.shape[1]
    s = np.cov(np.column_stack([y, x]), rowvar=False)
    var_y, cov_yx, cov_xx = s[0, 0], s[0, 1:], s[1:, 1:]

    # var(y - xw) = var_y - 2 w'cov_yx + w'cov_xx w: problema quadratico convesso
    def obiettivo(w):
        return var_y - 2 * w @ cov_yx + w @ cov_xx @ w

    def gradiente(w):
        return -2 * cov_yx + 2 * cov_xx @ w

    stima = minimize(obiettivo, np.full(k, 1 / k), jac=gradiente, method="SLSQP", bounds=[(0, 1)] * k,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(k)}],
                     options={"ftol": 1e-14, "maxiter": 500})
    w = np.clip(stima.x, 0, None)
    return w / w.sum()


def analisi_stile(rendimenti: pd.Series, indici: pd.DataFrame) -> Stile:
    """Stima i pesi di stile del fondo sull'intero periodo comune con gli indici."""
    dati = pd.concat([rendimenti.rename("_r"), indici], axis=1, join="inner").dropna()
    if len(dati) < len(indici.columns) + 12:
        raise ValueError("Troppi pochi mesi per la style analysis rispetto al numero di indici")
    y, x = dati["_r"].to_numpy(), dati[indici.columns].to_numpy()
    w = _pesi_stile(y, x)
    differenze = pd.Series(y - x @ w, index=dati.index, name="Selezione")
    return Stile(
        pesi=pd.Series(w, index=indici.columns),
        r2=float(1 - differenze.var() / np.var(y, ddof=1)),
        selezione=float(differenze.mean() * 12),
        tracking_error=float(differenze.std() * np.sqrt(12)),
        differenze=differenze,
    )


def stile_mobile(rendimenti: pd.Series, indici: pd.DataFrame, finestra: int = 36) -> pd.DataFrame:
    """Pesi di stile su finestre mobili di ``finestra`` mesi: rivelano i cambi di strategia del gestore."""
    dati = pd.concat([rendimenti.rename("_r"), indici], axis=1, join="inner").dropna()
    if len(dati) < finestra:
        raise ValueError(f"Servono almeno {finestra} mesi per la style analysis mobile")
    y, x = dati["_r"].to_numpy(), dati[indici.columns].to_numpy()
    righe = [_pesi_stile(y[t - finestra:t], x[t - finestra:t]) for t in range(finestra, len(dati) + 1)]
    return pd.DataFrame(righe, index=dati.index[finestra - 1:], columns=indici.columns)
