"""Attribuzione di performance di Brinson: allocazione, selezione e interazione per settore.

Per ogni periodo e segmento (settore, area geografica, asset class), con w pesi e r rendimenti:

- allocazione  = (w_fondo - w_bench) * (r_bench_segmento - r_bench_totale)   [Brinson-Fachler]
- selezione    = w_bench * (r_fondo_segmento - r_bench_segmento)
- interazione  = (w_fondo - w_bench) * (r_fondo_segmento - r_bench_segmento)

Gli effetti di un periodo sommano esattamente al rendimento attivo del periodo. Su più periodi i rendimenti
si compongono e la somma non torna più: gli effetti vengono collegati con il metodo logaritmico di Cariño,
che conserva la somma esatta con il rendimento attivo cumulato.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EFFETTI = ["Allocazione", "Selezione", "Interazione"]
COLONNE = ["data", "settore", "peso_fondo", "peso_benchmark", "rend_fondo", "rend_benchmark"]


@dataclass
class Attribuzione:
    """Attribuzione collegata sull'intero periodo e dettaglio per periodo."""

    per_segmento: pd.DataFrame  # effetti collegati per segmento, con pesi medi e rendimenti cumulati
    totale: pd.Series  # effetti collegati totali, rendimento del fondo, del benchmark e attivo
    per_periodo: pd.DataFrame  # effetti di ogni periodo (somma sui segmenti), non collegati
    cumulato: pd.DataFrame = field(repr=False)  # effetti collegati cumulati fino a ogni data
    annuale: pd.DataFrame = field(repr=False)  # effetti collegati per anno solare


def verifica(dati: pd.DataFrame, tolleranza: float = 1e-4) -> pd.DataFrame:
    """Controlla colonne e pesi (che devono sommare a 1 in ogni periodo) e ordina per data e segmento."""
    mancanti = [c for c in COLONNE if c not in dati]
    if mancanti:
        raise ValueError(f"Colonne mancanti: {', '.join(mancanti)}")
    d = dati[COLONNE].copy()
    d["data"] = pd.to_datetime(d["data"])
    somme = d.groupby("data")[["peso_fondo", "peso_benchmark"]].sum()
    errate = somme[(somme - 1).abs().gt(tolleranza).any(axis=1)]
    if not errate.empty:
        raise ValueError(f"In {len(errate)} periodi i pesi non sommano a 1 (primo: {errate.index[0]:%Y-%m})")
    if d.duplicated(["data", "settore"]).any():
        raise ValueError("Ci sono segmenti ripetuti nella stessa data")
    # Piccoli scarti di arrotondamento: i pesi vengono riportati a somma esattamente 1
    for c in ["peso_fondo", "peso_benchmark"]:
        d[c] = d[c] / d.groupby("data")[c].transform("sum")
    return d.sort_values(["data", "settore"]).reset_index(drop=True)


def effetti_periodo(d: pd.DataFrame, metodo: str = "fachler") -> pd.DataFrame:
    """Effetti di allocazione, selezione e interazione per ogni riga (periodo e segmento).

    ``metodo="fachler"`` (default) misura l'allocazione rispetto al rendimento totale del benchmark: un
    sovrappeso conta positivamente solo se il segmento ha fatto meglio dell'indice. ``metodo="bhb"`` usa la
    versione originale di Brinson, Hood e Beebower: il totale è identico, cambia la ripartizione tra segmenti.
    """
    if metodo not in ("fachler", "bhb"):
        raise ValueError("metodo deve essere 'fachler' o 'bhb'")
    d = d.copy()
    attivo = d["peso_fondo"] - d["peso_benchmark"]
    rb_tot = (d["peso_benchmark"] * d["rend_benchmark"]).groupby(d["data"]).transform("sum")
    riferimento = rb_tot if metodo == "fachler" else 0.0
    d["Allocazione"] = attivo * (d["rend_benchmark"] - riferimento)
    d["Selezione"] = d["peso_benchmark"] * (d["rend_fondo"] - d["rend_benchmark"])
    d["Interazione"] = attivo * (d["rend_fondo"] - d["rend_benchmark"])
    return d


def _coefficiente_carino(rp, rb):
    """k = (ln(1+rp) - ln(1+rb)) / (rp - rb), con limite 1/(1+r) quando rp = rb."""
    rp, rb = np.asarray(rp, dtype=float), np.asarray(rb, dtype=float)
    diff = rp - rb
    uguali = np.isclose(diff, 0, atol=1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(uguali, 1 / (1 + rp), (np.log1p(rp) - np.log1p(rb)) / np.where(uguali, 1, diff))
    return k


def _collega(eff: pd.DataFrame, rp: pd.Series, rb: pd.Series) -> tuple[pd.DataFrame, float, float]:
    """Collega gli effetti di più periodi con Cariño. Restituisce effetti per riga riscalati e rendimenti cumulati."""
    rp_cum, rb_cum = float((1 + rp).prod() - 1), float((1 + rb).prod() - 1)
    k_tot = float(_coefficiente_carino(rp_cum, rb_cum))
    k_t = pd.Series(_coefficiente_carino(rp, rb), index=rp.index)
    fattore = eff["data"].map(k_t / k_tot)
    collegati = eff[EFFETTI].mul(fattore, axis=0)
    return collegati, rp_cum, rb_cum


def attribuzione_brinson(dati: pd.DataFrame, metodo: str = "fachler") -> Attribuzione:
    """Attribuzione di Brinson su più periodi con collegamento di Cariño.

    ``dati`` è una tabella lunga con una riga per data e segmento e le colonne di :data:`COLONNE`: pesi a
    inizio periodo e rendimenti del periodo, del fondo e del benchmark, per ogni segmento. I rendimenti sono
    lordi di commissioni, perché misurano le scelte di gestione.
    """
    d = effetti_periodo(verifica(dati), metodo)
    rp = (d["peso_fondo"] * d["rend_fondo"]).groupby(d["data"]).sum()
    rb = (d["peso_benchmark"] * d["rend_benchmark"]).groupby(d["data"]).sum()

    collegati, rp_cum, rb_cum = _collega(d, rp, rb)
    collegati["settore"] = d["settore"]
    per_segmento = collegati.groupby("settore")[EFFETTI].sum()
    per_segmento["Totale"] = per_segmento.sum(axis=1)
    per_segmento.insert(0, "Peso medio fondo", d.groupby("settore")["peso_fondo"].mean())
    per_segmento.insert(1, "Peso medio benchmark", d.groupby("settore")["peso_benchmark"].mean())
    per_segmento.insert(2, "Rendimento fondo", d.groupby("settore")["rend_fondo"].apply(lambda x: (1 + x).prod() - 1))
    per_segmento.insert(3, "Rendimento benchmark",
                        d.groupby("settore")["rend_benchmark"].apply(lambda x: (1 + x).prod() - 1))
    per_segmento = per_segmento.sort_values("Totale", ascending=False)

    totale = per_segmento[EFFETTI].sum()
    totale["Rendimento attivo"] = rp_cum - rb_cum
    totale["Rendimento fondo"] = rp_cum
    totale["Rendimento benchmark"] = rb_cum

    per_periodo = d.groupby("data")[EFFETTI].sum()
    per_periodo["Rendimento fondo"], per_periodo["Rendimento benchmark"] = rp, rb

    # Effetti collegati dall'inizio fino a ogni data, per il grafico dell'evoluzione nel tempo
    righe = {}
    for fine in rp.index:
        sotto = d[d["data"] <= fine]
        c, a, b = _collega(sotto, rp[:fine], rb[:fine])
        righe[fine] = {**c.sum().to_dict(), "Rendimento attivo": a - b}
    cumulato = pd.DataFrame(righe).T

    annuale = {}
    for anno, sotto in d.groupby(d["data"].dt.year):
        date = sotto["data"].unique()
        c, a, b = _collega(sotto, rp[date], rb[date])
        annuale[anno] = {**c.sum().to_dict(), "Rendimento attivo": a - b, "Rendimento fondo": a,
                         "Rendimento benchmark": b}
    return Attribuzione(per_segmento, totale, per_periodo, cumulato, pd.DataFrame(annuale).T)


def active_share(pesi_fondo: pd.Series, pesi_benchmark: pd.Series) -> float:
    """Active share di Cremers e Petajisto: metà della somma degli scostamenti assoluti dei pesi.

    Calcolato sui segmenti (settori) è un limite inferiore di quello sui singoli titoli.
    """
    indice = pesi_fondo.index.union(pesi_benchmark.index)
    a = pesi_fondo.reindex(indice).fillna(0) - pesi_benchmark.reindex(indice).fillna(0)
    return float(a.abs().sum() / 2)
