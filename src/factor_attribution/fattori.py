"""Modelli fattoriali: CAPM, Fama-French a 3 e 5 fattori, Carhart, con errori robusti e bootstrap dell'alfa.

Tutte le regressioni sono sui rendimenti mensili in eccesso rispetto al tasso privo di rischio:

    r_t - rf_t = alfa + somma_k beta_k * f_k,t + e_t
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

MODELLI = {
    "CAPM": ["MKT"],
    "Fama-French 3": ["MKT", "SMB", "HML"],
    "Carhart 4": ["MKT", "SMB", "HML", "MOM"],
    "Fama-French 5": ["MKT", "SMB", "HML", "RMW", "CMA"],
    "Fama-French 5 + Momentum": ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"],
}
MODELLO_COMPLETO = "Fama-French 5 + Momentum"


@dataclass
class Regressione:
    """Risultato di una regressione fattoriale. Alfa e volatilità sono annualizzati."""

    modello: str
    fattori: list[str]
    alfa: float  # alfa annuo (alfa mensile x 12)
    alfa_t: float  # statistica t con errori di Newey-West
    alfa_p: float
    beta: pd.Series
    beta_errore: pd.Series
    beta_t: pd.Series
    beta_p: pd.Series
    r2: float
    r2_corretto: float
    bic: float
    volatilita_residua: float  # deviazione standard annua dei residui
    n: int
    residui: pd.Series = field(repr=False)
    eccesso: pd.Series = field(repr=False)  # rendimenti in eccesso del fondo
    dati_fattori: pd.DataFrame = field(repr=False)  # fattori allineati alle date del fondo

    @property
    def appraisal_ratio(self) -> float:
        """Alfa diviso rischio specifico: quanto alfa per unità di rischio non spiegato dai fattori."""
        return self.alfa / self.volatilita_residua if self.volatilita_residua > 0 else np.nan

    def intervallo_beta(self, livello: float = 0.95) -> pd.DataFrame:
        from scipy import stats

        z = stats.norm.ppf(0.5 + livello / 2)
        return pd.DataFrame({"min": self.beta - z * self.beta_errore, "max": self.beta + z * self.beta_errore})


def allinea(rendimenti: pd.Series, fattori: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Allinea fondo e fattori sulle date comuni e restituisce il rendimento in eccesso e i fattori."""
    dati = pd.concat([rendimenti.rename("_r"), fattori], axis=1, join="inner").dropna()
    if len(dati) < 24:
        raise ValueError(f"Servono almeno 24 mesi in comune tra fondo e fattori (ora {len(dati)})")
    return (dati["_r"] - dati["RF"]).rename(rendimenti.name), dati.drop(columns="_r")


def lag_newey_west(n: int) -> int:
    """Numero di ritardi di Newey-West con la regola di Newey e West (1994)."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def regressione(
    rendimenti: pd.Series, fattori: pd.DataFrame, modello: str | list[str] = MODELLO_COMPLETO
) -> Regressione:
    """Regressione dei rendimenti mensili in eccesso del fondo sui fattori del modello indicato.

    ``modello`` è il nome di uno dei :data:`MODELLI` oppure un elenco di colonne di ``fattori``.

    Gli errori standard sono di Newey-West (robusti a eteroschedasticità e autocorrelazione), perché i
    rendimenti dei fondi hanno spesso volatilità variabile e una lieve autocorrelazione.
    """
    nomi = list(modello) if not isinstance(modello, str) else MODELLI[modello]
    modello = modello if isinstance(modello, str) else " + ".join(nomi)
    mancanti = [f for f in nomi if f not in fattori]
    if mancanti:
        raise ValueError(f"Fattori mancanti per il modello {modello}: {', '.join(mancanti)}")
    y, f = allinea(rendimenti, fattori)
    x = sm.add_constant(f[nomi])
    stima = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag_newey_west(len(y))})
    residui = stima.resid
    return Regressione(
        modello=modello,
        fattori=nomi,
        alfa=float(stima.params["const"] * 12),
        alfa_t=float(stima.tvalues["const"]),
        alfa_p=float(stima.pvalues["const"]),
        beta=stima.params[nomi],
        beta_errore=stima.bse[nomi],
        beta_t=stima.tvalues[nomi],
        beta_p=stima.pvalues[nomi],
        r2=float(stima.rsquared),
        r2_corretto=float(stima.rsquared_adj),
        bic=float(stima.bic),
        volatilita_residua=float(residui.std(ddof=len(nomi) + 1) * np.sqrt(12)),
        n=int(stima.nobs),
        residui=residui,
        eccesso=y,
        dati_fattori=f,
    )


def confronta_modelli(rendimenti: pd.Series, fattori: pd.DataFrame) -> pd.DataFrame:
    """Alfa, significatività e capacità esplicativa di ogni modello disponibile con i fattori forniti."""
    righe = {}
    for nome, elenco in MODELLI.items():
        if not all(f in fattori for f in elenco):
            continue
        r = regressione(rendimenti, fattori, nome)
        righe[nome] = {"Alfa annuo": r.alfa, "t alfa": r.alfa_t, "p-value alfa": r.alfa_p,
                       "Beta mercato": r.beta["MKT"], "R²": r.r2, "R² corretto": r.r2_corretto,
                       "BIC": r.bic, "Rischio specifico": r.volatilita_residua}
    return pd.DataFrame(righe).T


def scomposizione_rendimento(reg: Regressione) -> pd.Series:
    """Rendimento medio annuo del fondo scomposto in tasso privo di rischio, premi dei fattori e alfa.

    Con l'intercetta i residui hanno media zero, quindi la somma è esattamente il rendimento medio aritmetico
    annualizzato del fondo: media(r) = media(rf) + alfa + somma_k beta_k * media(f_k).
    """
    f = reg.dati_fattori
    parti = {"RF": float(f["RF"].mean() * 12)}
    for k in reg.fattori:
        parti[k] = float(reg.beta[k] * f[k].mean() * 12)
    parti["Alfa"] = reg.alfa
    return pd.Series(parti)


def scomposizione_rischio(reg: Regressione) -> pd.Series:
    """Quota della varianza del fondo dovuta a ogni fattore (scomposizione di Eulero) e al rischio specifico.

    La quota del fattore k è beta_k * cov(f_k, componente sistematica) / var(r): le quote sommano a 1 e
    possono essere negative per un fattore che riduce il rischio complessivo.
    """
    f = reg.dati_fattori[reg.fattori]
    sistematica = f @ reg.beta
    totale = reg.eccesso.var(ddof=0)
    quote = {k: float(reg.beta[k] * np.cov(f[k], sistematica, ddof=0)[0, 1] / totale) for k in reg.fattori}
    quote["Specifico"] = float(reg.residui.var(ddof=0) / totale)
    return pd.Series(quote)


def esposizioni_mobili(
    rendimenti: pd.Series, fattori: pd.DataFrame, modello: str = MODELLO_COMPLETO, finestra: int = 36
) -> pd.DataFrame:
    """Beta e alfa annuo stimati su finestre mobili di ``finestra`` mesi: mostrano se lo stile cambia nel tempo."""
    nomi = MODELLI[modello]
    y, f = allinea(rendimenti, fattori)
    if len(y) < finestra + 1:
        raise ValueError(f"Servono più di {finestra} mesi per le stime mobili")
    stima = RollingOLS(y, sm.add_constant(f[nomi]), window=finestra).fit()
    p = stima.params.dropna()
    p["const"] *= 12
    return p.rename(columns={"const": "Alfa"})


def bootstrap_alfa(
    reg: Regressione, n: int = 5000, blocco: int = 6, seed: int = 0
) -> dict[str, float | np.ndarray]:
    """Talento o fortuna? Bootstrap dell'alfa sotto l'ipotesi che il gestore non abbia capacità.

    Si costruiscono rendimenti artificiali con le stesse esposizioni ai fattori e alfa nullo, ricampionando a
    blocchi (per conservare l'autocorrelazione) i mesi di fattori e residui insieme, e si conta quante volte un
    gestore senza capacità ottiene per caso una statistica t dell'alfa almeno pari a quella osservata
    (approccio di Kosowski, Timmermann, Wermers e White, 2006).

    Restituisce il p-value a una coda nella direzione dell'alfa osservato, quello a due code, la soglia che un
    gestore senza capacità supera per caso nel 5% dei casi e la distribuzione simulata della statistica t.
    """
    rng = np.random.default_rng(seed)
    x = sm.add_constant(reg.dati_fattori[reg.fattori]).to_numpy()
    e = reg.residui.to_numpy()
    beta = np.r_[0.0, reg.beta.to_numpy()]
    m = len(e)
    blocchi = int(np.ceil(m / blocco))

    def t_alfa(xs: np.ndarray, ys: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(xs, ys, rcond=None)
        res = ys - xs @ coef
        s2 = res @ res / (m - xs.shape[1])
        return float(coef[0] / np.sqrt(s2 * np.linalg.inv(xs.T @ xs)[0, 0]))

    # La statistica osservata è calcolata come quelle simulate, per un confronto omogeneo
    t = t_alfa(x, reg.eccesso.to_numpy())
    t_simulati = np.empty(n)
    for i in range(n):
        inizi = rng.integers(0, m, blocchi)
        idx = ((inizi[:, None] + np.arange(blocco)) % m).ravel()[:m]  # blocchi circolari
        t_simulati[i] = t_alfa(x[idx], x[idx] @ beta + e[idx])
    una_coda = np.mean(t_simulati >= t) if t >= 0 else np.mean(t_simulati <= t)
    return {"t osservato": t, "p-value una coda": float(una_coda),
            "p-value due code": float(np.mean(np.abs(t_simulati) >= abs(t))),
            "t al 95% per fortuna": float(np.quantile(t_simulati, 0.95)), "t simulati": t_simulati}
