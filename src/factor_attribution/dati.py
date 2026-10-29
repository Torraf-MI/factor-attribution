"""Dati: fattori di rischio, rendimenti dei fondi, indici di stile e pesi per settore.

Le fonti sono file CSV, la libreria di Kenneth French (fattori reali, scaricati da internet) oppure un
universo dimostrativo sintetico, costruito in modo coerente: i fondi sono portafogli di settori, e i settori
rispondono agli stessi fattori usati nelle regressioni.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FATTORI = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

NOMI_FATTORI = {
    "MKT": "Mercato",
    "SMB": "Small cap",
    "HML": "Value",
    "RMW": "Qualità (redditività)",
    "CMA": "Investimenti prudenti",
    "MOM": "Momentum",
    "RF": "Tasso privo di rischio",
}

# Fattori mensili dimostrativi: (media, volatilità) mensili dei premi. Ordini di grandezza plausibili
# per un mercato azionario sviluppato, non stime storiche.
PARAMETRI_FATTORI = {
    "MKT": (0.0065, 0.045),
    "SMB": (0.0010, 0.022),
    "HML": (0.0030, 0.028),
    "RMW": (0.0025, 0.016),
    "CMA": (0.0012, 0.018),
    "MOM": (0.0050, 0.035),
}

CORRELAZIONE_FATTORI = np.array(
    [
        # MKT   SMB   HML   RMW   CMA   MOM
        [1.00, 0.15, 0.20, -0.30, -0.25, -0.20],
        [0.15, 1.00, 0.05, -0.30, 0.00, -0.05],
        [0.20, 0.05, 1.00, -0.10, 0.55, -0.40],
        [-0.30, -0.30, -0.10, 1.00, 0.10, 0.15],
        [-0.25, 0.00, 0.55, 0.10, 1.00, 0.05],
        [-0.20, -0.05, -0.40, 0.15, 0.05, 1.00],
    ]
)

# Tasso privo di rischio annuo in euro: punti di riferimento interpolati linearmente
TASSO_EURO = {
    "2010-01": 0.006, "2011-07": 0.012, "2012-07": 0.001, "2014-09": -0.001, "2016-03": -0.003,
    "2019-09": -0.005, "2022-06": -0.005, "2022-12": 0.019, "2023-09": 0.039, "2024-06": 0.037,
    "2025-06": 0.020, "2025-12": 0.020,
}

# Mesi con movimenti eccezionali (rendimenti mensili dei premi), simili per segno e ordine di grandezza a
# episodi reali: servono a dare ai dati le code pesanti e le rotazioni di stile dei mercati veri.
EVENTI = {
    "2011-08": {"MKT": -0.11, "HML": -0.03, "MOM": 0.03},
    "2015-08": {"MKT": -0.08},
    "2018-12": {"MKT": -0.07},
    "2020-02": {"MKT": -0.09},
    "2020-03": {"MKT": -0.15, "HML": -0.08, "SMB": -0.05},
    "2020-11": {"MKT": 0.14, "HML": 0.09, "SMB": 0.04, "MOM": -0.12},
    "2022-03": {"HML": 0.06, "MOM": 0.03},
    "2022-09": {"MKT": -0.08},
}

# Settori dell'indice azionario: peso iniziale nel benchmark, esposizioni ai fattori
# (MKT, SMB, HML, RMW, CMA, MOM) e volatilità specifica mensile.
SETTORI = {
    "Finanziari": (0.19, (1.15, 0.00, 0.70, -0.10, 0.00, -0.10), 0.0150),
    "Industriali": (0.16, (1.10, 0.20, 0.10, 0.10, 0.00, 0.05), 0.0100),
    "Sanità": (0.14, (0.75, -0.10, -0.30, 0.30, 0.10, 0.05), 0.0150),
    "Consumi discrezionali": (0.10, (1.15, 0.15, 0.00, 0.00, -0.20, 0.00), 0.0150),
    "Beni di prima necessità": (0.10, (0.60, -0.15, -0.10, 0.40, 0.20, 0.00), 0.0125),
    "Tecnologia": (0.08, (1.25, 0.10, -0.60, 0.10, -0.40, 0.15), 0.0175),
    "Materiali": (0.06, (1.10, 0.10, 0.40, -0.10, 0.10, -0.05), 0.0175),
    "Energia": (0.06, (0.95, 0.00, 0.70, 0.00, 0.20, 0.00), 0.0250),
    "Utilities": (0.05, (0.60, -0.10, 0.30, -0.10, 0.40, 0.00), 0.0150),
    "Comunicazioni": (0.04, (0.80, 0.00, 0.10, 0.00, 0.00, -0.10), 0.0150),
    "Immobiliare": (0.02, (0.90, 0.30, 0.30, -0.20, 0.20, -0.10), 0.0200),
}

# Fondi dimostrativi: sovrappesi per settore rispetto al benchmark, esposizioni aggiuntive ai fattori
# ottenute con la scelta dei titoli dentro i settori, alfa vero lordo mensile, volatilità della selezione
# per settore e costo annuo (TER).
FONDI = {
    "Fondo Europa Value Attivo": {
        "sovrappesi": {"Finanziari": 0.06, "Energia": 0.04, "Materiali": 0.02, "Utilities": 0.01,
                       "Tecnologia": -0.05, "Sanità": -0.04, "Beni di prima necessità": -0.03,
                       "Consumi discrezionali": -0.01},
        "fattori": {"HML": 0.35, "SMB": 0.15},
        "alfa": 0.0, "selezione": 0.015, "ter": 0.0165,
    },
    "Fondo Europa Qualità": {
        "sovrappesi": {"Beni di prima necessità": 0.05, "Sanità": 0.04, "Tecnologia": 0.02,
                       "Finanziari": -0.06, "Energia": -0.03, "Materiali": -0.02},
        "fattori": {"RMW": 0.30, "MOM": 0.05},
        "alfa": 0.0025, "selezione": 0.012, "ter": 0.0130,
    },
    "Fondo Europa Large Cap": {
        "sovrappesi": {}, "fattori": {}, "alfa": 0.0, "selezione": 0.004, "ter": 0.0175,
    },
    "ETF Azionario Europa": {
        "sovrappesi": {}, "fattori": {}, "alfa": 0.0, "selezione": 0.0004, "ter": 0.0015,
    },
}

FONDO_ESEMPIO = "Fondo Europa Value Attivo"
BENCHMARK_ESEMPIO = "Indice Azionario Europa"

# Indici di stile per la returns-based style analysis: esposizioni ai fattori e volatilità specifica
INDICI_STILE = {
    "Azioni Europa Value": ((1.00, 0.10, 0.55, 0.00, 0.10, -0.10), 0.008),
    "Azioni Europa Growth": ((1.00, -0.05, -0.45, 0.15, -0.15, 0.10), 0.008),
    "Azioni Europa Small Cap": ((1.05, 0.85, 0.10, -0.10, 0.00, 0.00), 0.010),
    "Azioni Mondo": ((0.85, 0.00, 0.00, 0.05, 0.00, 0.00), 0.020),
}

TER_ESEMPIO = {nome: f["ter"] for nome, f in FONDI.items()}


def _mesi(inizio: str, fine: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(inizio), pd.Timestamp(fine) + pd.offsets.MonthEnd(0), freq="ME")


def tasso_privo_di_rischio(date: pd.DatetimeIndex) -> pd.Series:
    """Tasso mensile in euro dimostrativo, interpolato tra i punti di :data:`TASSO_EURO`."""
    punti = pd.Series(TASSO_EURO)
    punti.index = pd.to_datetime(punti.index) + pd.offsets.MonthEnd(0)
    annuo = punti.reindex(punti.index.union(date)).interpolate(method="time").ffill().bfill().reindex(date)
    return (annuo / 12).rename("RF")


def generate_sample_factors(inizio: str = "2010-01", fine: str = "2025-12", seed: int = 7) -> pd.DataFrame:
    """Premi mensili dei sei fattori e tasso privo di rischio, sintetici.

    Code pesanti (t di Student con 5 gradi di libertà), correlazioni tra fattori e alcuni mesi eccezionali.
    Media e volatilità di ogni premio sul campione sono esattamente quelle di :data:`PARAMETRI_FATTORI`,
    qualunque sia il seme casuale. Non sono i fattori storici di alcun mercato.
    """
    rng = np.random.default_rng(seed)
    date = _mesi(inizio, fine)
    n, k = len(date), len(FATTORI)
    gradi = 5
    z = rng.multivariate_normal(np.zeros(k), CORRELAZIONE_FATTORI, size=n)
    z *= np.sqrt((gradi - 2) / rng.chisquare(gradi, size=n))[:, None]
    f = pd.DataFrame(z, index=date, columns=FATTORI)
    for mese, shock in EVENTI.items():
        quando = date == pd.Timestamp(mese) + pd.offsets.MonthEnd(0)
        for fattore, valore in shock.items():
            # In unità di deviazioni standard, per il riscalamento successivo
            f.loc[quando, fattore] = valore / PARAMETRI_FATTORI[fattore][1]
    medie = np.array([m for m, _ in PARAMETRI_FATTORI.values()])
    vol = np.array([v for _, v in PARAMETRI_FATTORI.values()])
    f = (f - f.mean()) / f.std() * vol + medie
    f["RF"] = tasso_privo_di_rischio(date)
    return f.round(6)


def _esposizione(fattori: pd.DataFrame, beta: tuple[float, ...] | dict[str, float]) -> pd.Series:
    b = pd.Series(beta, index=FATTORI if not isinstance(beta, dict) else None, dtype=float)
    b = b.reindex(FATTORI).fillna(0.0)
    return fattori[FATTORI] @ b


def generate_sample_universe(seed: int = 11) -> dict[str, pd.DataFrame]:
    """Universo dimostrativo coerente: fattori, settori, fondi, benchmark e indici di stile.

    Restituisce un dizionario con:

    - ``fattori``: premi mensili e tasso privo di rischio (:func:`generate_sample_factors`);
    - ``fondi``: rendimenti mensili netti dei fondi e del benchmark;
    - ``settori``: tabella lunga con pesi e rendimenti lordi per settore del fondo di esempio e del
      benchmark, per l'attribuzione di Brinson;
    - ``indici``: rendimenti mensili degli indici di stile, per la style analysis.
    """
    rng = np.random.default_rng(seed)
    fattori = generate_sample_factors()
    date = fattori.index
    rf = fattori["RF"]
    n, nomi = len(date), list(SETTORI)

    # Rendimenti dei settori del benchmark: tasso privo di rischio + esposizioni ai fattori + parte specifica
    rend_b = pd.DataFrame(
        {s: rf + _esposizione(fattori, beta) + rng.normal(0, vol, n) for s, (_, beta, vol) in SETTORI.items()},
        index=date,
    )

    # Pesi del benchmark a capitalizzazione: derivano con i rendimenti, con un lieve ritorno verso i pesi base
    base = np.array([p for p, _, _ in SETTORI.values()])
    pesi_b = np.empty((n, len(nomi)))
    w = base.copy()
    for t in range(n):
        pesi_b[t] = w
        w = w * (1 + rend_b.iloc[t].to_numpy())
        w = 0.97 * w / w.sum() + 0.03 * base
    pesi_b = pd.DataFrame(pesi_b, index=date, columns=nomi)
    benchmark = (pesi_b * rend_b).sum(axis=1)

    fondi, settori = {}, None
    for nome, f in FONDI.items():
        sovrappesi = pd.Series(f["sovrappesi"], dtype=float).reindex(nomi).fillna(0.0)
        # Scostamenti tattici persistenti attorno ai sovrappesi strategici
        rumore = np.zeros((n, len(nomi)))
        for t in range(1, n):
            rumore[t] = 0.9 * rumore[t - 1] + rng.normal(0, 0.003 if f["sovrappesi"] else 0.0005, len(nomi))
        wp = np.clip(pesi_b.to_numpy() + sovrappesi.to_numpy() + rumore, 0.0, None)
        wp = pd.DataFrame(wp / wp.sum(axis=1, keepdims=True), index=date, columns=nomi)
        extra = _esposizione(fattori, f["fattori"]) if f["fattori"] else pd.Series(0.0, index=date)
        rend_p = rend_b.add(extra + f["alfa"], axis=0) + rng.normal(0, f["selezione"], (n, len(nomi)))
        lordo = (wp * rend_p).sum(axis=1)
        fondi[nome] = lordo - f["ter"] / 12
        if nome == FONDO_ESEMPIO:
            settori = pd.DataFrame({
                "data": np.repeat(date, len(nomi)),
                "settore": np.tile(nomi, n),
                "peso_fondo": wp.to_numpy().ravel(),
                "peso_benchmark": pesi_b.to_numpy().ravel(),
                "rend_fondo": rend_p.to_numpy().ravel(),
                "rend_benchmark": rend_b.to_numpy().ravel(),
            })

    fondi = pd.DataFrame(fondi)
    fondi[BENCHMARK_ESEMPIO] = benchmark

    indici = {s: rf + _esposizione(fattori, beta) + rng.normal(0, vol, n) for s, (beta, vol) in INDICI_STILE.items()}
    indici.update(_obbligazioni(fattori, rng))
    indici["Monetario EUR"] = rf
    indici = pd.DataFrame(indici)

    return {
        "fattori": fattori,
        "fondi": fondi.round(6),
        "settori": settori.round({c: 6 for c in ["peso_fondo", "peso_benchmark", "rend_fondo", "rend_benchmark"]}),
        "indici": indici.round(6),
    }


def _obbligazioni(fattori: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.Series]:
    """Indici obbligazionari in euro con il ribasso del 2022 dovuto al rialzo dei tassi."""
    date, rf = fattori.index, fattori["RF"]
    n = len(date)
    gov = rf + 0.0022 + rng.normal(0, 0.012, n) - 0.03 * fattori["MKT"].to_numpy()
    anno_2022 = (date.year == 2022).astype(float)
    gov = gov - 0.016 * anno_2022
    corp = rf + 0.0018 + 0.6 * (gov - rf) + 0.12 * fattori["MKT"] + rng.normal(0, 0.005, n) - 0.004 * anno_2022
    return {"Obbligazioni Gov. EUR": pd.Series(gov, index=date), "Obbligazioni Corporate EUR": corp}


def fondo_bilanciato_esempio(indici: pd.DataFrame, seed: int = 5) -> pd.Series:
    """Fondo bilanciato flessibile dimostrativo che aumenta l'azionario nel tempo (deriva di stile).

    L'azionario passa dal 30% al 60%, con un cambio di stile da value a growth a metà periodo: serve a mostrare
    come la style analysis mobile riconosce la strategia effettiva del gestore dai soli rendimenti.
    """
    rng = np.random.default_rng(seed)
    n = len(indici)
    t = np.linspace(0, 1, n)
    azioni = 0.30 + 0.30 * t
    quota_value = np.where(t < 0.5, 0.7, 0.3)
    pesi = pd.DataFrame({
        "Azioni Europa Value": azioni * quota_value * 0.7,
        "Azioni Europa Growth": azioni * (1 - quota_value) * 0.7,
        "Azioni Mondo": azioni * 0.3,
        "Obbligazioni Gov. EUR": (1 - azioni) * 0.55,
        "Obbligazioni Corporate EUR": (1 - azioni) * 0.35,
        "Monetario EUR": (1 - azioni) * 0.10,
    }, index=indici.index)
    lordo = (pesi * indici[pesi.columns]).sum(axis=1) + rng.normal(0, 0.003, n)
    return (lordo - 0.019 / 12).rename("Fondo Bilanciato Flessibile")


# ---------------------------------------------------------------------------------------------------------
# Caricamento da file e da internet
# ---------------------------------------------------------------------------------------------------------

def load_csv(path: str | Path, date_column: str | None = None) -> pd.DataFrame:
    """Legge un CSV con una colonna data e una colonna numerica per ogni serie."""
    df = pd.read_csv(path)
    date_column = date_column or df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column])
    return df.set_index(date_column).sort_index().astype(float)


def sono_prezzi(df: pd.DataFrame) -> bool:
    """True se le colonne sembrano prezzi o valori quota (tutti positivi e lontani da zero)."""
    valori = df.to_numpy(dtype=float)
    valori = valori[~np.isnan(valori)]
    return bool(valori.size and (valori > 0).all() and np.median(np.abs(valori)) > 2)


def rendimenti_mensili(df: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti mensili da prezzi giornalieri o mensili; se ``df`` contiene già rendimenti li allinea a fine mese.

    I rendimenti sono in forma decimale (0,012 = 1,2%).
    """
    if sono_prezzi(df):
        mensili = df.ffill().resample("ME").last()
        return mensili.pct_change(fill_method=None).iloc[1:]
    r = df.copy()
    r.index = r.index + pd.offsets.MonthEnd(0)
    return r.groupby(level=0).apply(lambda g: (1 + g).prod(min_count=1) - 1)


def load_fattori(path: str | Path) -> pd.DataFrame:
    """Legge un CSV di fattori mensili (colonne MKT, SMB, HML, RMW, CMA, MOM, RF o i nomi di Kenneth French).

    Valori in decimali o in percentuale: se la volatilità del mercato supera 0,2 sono considerati percentuali.
    """
    f = _rinomina_french(load_csv(path))
    if "MKT" not in f or "RF" not in f:
        raise ValueError("Servono almeno le colonne del mercato (MKT o Mkt-RF) e del tasso privo di rischio (RF)")
    if f["MKT"].std() > 0.2:
        f = f / 100
    f.index = f.index + pd.offsets.MonthEnd(0)
    return f


def _rinomina_french(f: pd.DataFrame) -> pd.DataFrame:
    mappa = {"mkt-rf": "MKT", "mkt_rf": "MKT", "mkt": "MKT", "smb": "SMB", "hml": "HML", "rmw": "RMW",
             "cma": "CMA", "mom": "MOM", "wml": "MOM", "umd": "MOM", "rf": "RF"}
    return f.rename(columns=lambda c: mappa.get(str(c).strip().lower(), str(c).strip()))


def parse_french_csv(testo: str) -> pd.DataFrame:
    """Estrae la tabella mensile da un CSV della Kenneth French Data Library.

    I file hanno righe descrittive in testa, poi la tabella mensile (date AAAAMM, valori in percentuale) e dopo
    una tabella annuale: si legge solo la prima. Restituisce valori decimali con indice a fine mese.
    """
    righe = testo.splitlines()
    intestazione, dati = None, []
    for riga in righe:
        celle = [c.strip() for c in riga.split(",")]
        if intestazione is None:
            if len(celle) > 1 and celle[0] == "" and any(c for c in celle[1:]):
                intestazione = celle[1:]
            continue
        if re.fullmatch(r"\d{6}", celle[0]):
            dati.append(celle)
        elif dati:
            break  # fine della tabella mensile
    if intestazione is None or not dati:
        raise ValueError("Formato non riconosciuto: nessuna tabella mensile trovata")
    df = pd.DataFrame([[float(x) for x in r[1:len(intestazione) + 1]] for r in dati], columns=intestazione)
    df.index = pd.to_datetime([r[0] for r in dati], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.replace([-99.99, -999], np.nan) / 100
    return _rinomina_french(df)


URL_FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _scarica_zip(nome: str) -> str:  # pragma: no cover - richiede la rete
    with urllib.request.urlopen(f"{URL_FRENCH}{nome}_CSV.zip", timeout=30) as risposta:
        contenuto = risposta.read()
    with zipfile.ZipFile(io.BytesIO(contenuto)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def scarica_fattori_french(regione: str = "Europe") -> pd.DataFrame:  # pragma: no cover - richiede la rete
    """Scarica i cinque fattori di Fama-French e il momentum dalla Kenneth French Data Library.

    ``regione``: ``Europe``, ``Developed`` o ``US``. I fattori internazionali sono calcolati in dollari: per un
    investitore in euro i premi long-short (SMB, HML, RMW, CMA, MOM) restano confrontabili, mentre il mercato e
    il tasso privo di rischio includono l'effetto del cambio.
    """
    if regione == "US":
        cinque, momentum = "F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor"
    else:
        cinque, momentum = f"{regione}_5_Factors", f"{regione}_Mom_Factor"
    f = parse_french_csv(_scarica_zip(cinque))
    m = parse_french_csv(_scarica_zip(momentum))
    return f.join(m[["MOM"]], how="inner")
