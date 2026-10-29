"""Analisi completa di un fondo, grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from . import brinson, fattori, metriche, stile
from .dati import NOMI_FATTORI

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORI = [BLU_SCURO, ARANCIO, ACQUA, VIOLA, GIALLO, "#d14d9c", GRIGIO, BLU]
COLORI_FATTORI = dict(zip(["MKT", "SMB", "HML", "RMW", "CMA", "MOM"], COLORI))
COLORI_EFFETTI = {"Allocazione": BLU_SCURO, "Selezione": ARANCIO, "Interazione": GRIGIO}
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def _layout(fig: go.Figure, titolo: str, altezza: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza,
        template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO},
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


def _nome(f: str) -> str:
    return NOMI_FATTORI.get(f, f)


@dataclass
class Analisi:
    """Risultato completo dell'analisi di un fondo, usato per grafici e report."""

    fondo: str
    rendimenti: pd.Series  # rendimenti mensili netti del fondo
    benchmark: pd.Series | None
    nome_benchmark: str | None
    fattori: pd.DataFrame
    ter: float | None
    regressione: fattori.Regressione
    modelli: pd.DataFrame
    rendimento_fattori: pd.Series
    rischio_fattori: pd.Series
    mobili: pd.DataFrame
    bootstrap: dict = field(repr=False)
    metriche: dict[str, float] | None = None
    stile: stile.Stile | None = None
    stile_mobile: pd.DataFrame | None = None
    attribuzione: brinson.Attribuzione | None = None
    active_share: float | None = None
    mesi_attribuzione: int | None = None
    confronto: pd.DataFrame | None = field(default=None, repr=False)


def analizza(
    rendimenti: pd.Series,
    fattori_df: pd.DataFrame,
    benchmark: pd.Series | None = None,
    indici: pd.DataFrame | None = None,
    settori: pd.DataFrame | None = None,
    altri_fondi: pd.DataFrame | None = None,
    ter: float | None = None,
    modello: str = fattori.MODELLO_COMPLETO,
    finestra: int = 36,
    mesi_attribuzione: int | None = 36,
    n_bootstrap: int = 5000,
) -> Analisi:
    """Esegue tutte le analisi del report sul fondo indicato.

    ``rendimenti`` sono i rendimenti mensili netti del fondo; ``indici`` (facoltativi) abilitano la style
    analysis; ``settori`` (facoltativi, lordi di costi) l'attribuzione di Brinson sugli ultimi
    ``mesi_attribuzione`` mesi (``None`` per l'intero periodo); ``altri_fondi`` il confronto tra fondi.
    """
    nome = rendimenti.name or "Fondo"
    reg = fattori.regressione(rendimenti, fattori_df, modello)
    rf = fattori_df["RF"]

    mobili = (fattori.esposizioni_mobili(rendimenti, fattori_df, modello, finestra)
              if reg.n > finestra + 12 else pd.DataFrame())
    st = st_mobile = None
    if indici is not None:
        st = stile.analisi_stile(rendimenti, indici)
        if reg.n > finestra + 12:
            st_mobile = stile.stile_mobile(rendimenti, indici, finestra)

    attr = a_share = None
    if settori is not None:
        s = settori.copy()
        s["data"] = pd.to_datetime(s["data"])
        if mesi_attribuzione:
            date = sorted(s["data"].unique())[-mesi_attribuzione:]
            s = s[s["data"].isin(date)]
        attr = brinson.attribuzione_brinson(s)
        ultimo = s[s["data"] == s["data"].max()].set_index("settore")
        a_share = brinson.active_share(ultimo["peso_fondo"], ultimo["peso_benchmark"])

    confronto = None
    if altri_fondi is not None:
        tutti = altri_fondi.copy()
        if nome not in tutti:
            tutti[nome] = rendimenti
        righe = {}
        for c in tutti:
            r = tutti[c].dropna()
            capm = fattori.regressione(r, fattori_df, "CAPM")
            completo = fattori.regressione(r, fattori_df, modello)
            riga = {"Alfa CAPM": capm.alfa, "t CAPM": capm.alfa_t, "Alfa multifattore": completo.alfa,
                    "t multifattore": completo.alfa_t, "R²": completo.r2,
                    **{f"Beta {k}": completo.beta[k] for k in completo.fattori}}
            if benchmark is not None:
                riga.update(metriche.sintesi(r, benchmark, rf))
            else:
                riga.update({"Rendimento annuo": metriche.rendimento_annuo(r),
                             "Volatilità": metriche.volatilita(r)})
            righe[c] = riga
        confronto = pd.DataFrame(righe).T

    return Analisi(
        fondo=nome,
        rendimenti=rendimenti.dropna(),
        benchmark=benchmark,
        nome_benchmark=benchmark.name if benchmark is not None else None,
        fattori=fattori_df,
        ter=ter,
        regressione=reg,
        modelli=fattori.confronta_modelli(rendimenti, fattori_df),
        rendimento_fattori=fattori.scomposizione_rendimento(reg),
        rischio_fattori=fattori.scomposizione_rischio(reg),
        mobili=mobili,
        bootstrap=fattori.bootstrap_alfa(reg, n=n_bootstrap),
        metriche=metriche.sintesi(rendimenti.dropna(), benchmark, rf) if benchmark is not None else None,
        stile=st,
        stile_mobile=st_mobile,
        attribuzione=attr,
        active_share=a_share,
        mesi_attribuzione=attr.per_periodo.shape[0] if attr is not None else None,
        confronto=confronto,
    )


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def grafico_crescita(a: Analisi) -> go.Figure:
    r, b = a.rendimenti, a.benchmark
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=100 * (1 + r).cumprod(), name=a.fondo, line={"color": BLU_SCURO, "width": 2},
                             hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"))
    if b is not None:
        b = b.reindex(r.index)
        fig.add_trace(go.Scatter(x=b.index, y=100 * (1 + b).cumprod(), name=a.nome_benchmark,
                                 line={"color": GRIGIO, "width": 2, "dash": "dot"},
                                 hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"))
    return _layout(fig, "Valore di 100 investiti, al netto dei costi")


def grafico_modelli(a: Analisi) -> go.Figure:
    m = a.modelli
    errore = (m["Alfa annuo"] / m["t alfa"]).abs() * 1.96
    colori = [ACQUA if a_ > 0 else ROSSO for a_ in m["Alfa annuo"]]
    fig = go.Figure(go.Bar(
        x=m.index, y=m["Alfa annuo"], marker_color=colori,
        error_y={"type": "data", "array": errore, "color": TESTO_SECONDARIO, "thickness": 1.2},
        customdata=np.c_[m["t alfa"], m["R² corretto"]],
        hovertemplate="%{x}<br>Alfa %{y:.2%}<br>t = %{customdata[0]:.2f}<br>R² corretto %{customdata[1]:.3f}"
                      "<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".1%", title="Alfa annuo")
    return _layout(fig, "Alfa annuo per modello, con intervallo di confidenza al 95%", 400)


def grafico_esposizioni(a: Analisi) -> go.Figure:
    reg = a.regressione
    ic = reg.intervallo_beta()
    nomi = [_nome(f) for f in reg.fattori]
    fig = go.Figure(go.Bar(
        y=nomi, x=reg.beta, orientation="h", marker_color=[COLORI_FATTORI.get(f, BLU) for f in reg.fattori],
        error_x={"type": "data", "symmetric": False, "array": ic["max"] - reg.beta,
                 "arrayminus": reg.beta - ic["min"], "color": TESTO_SECONDARIO, "thickness": 1.2},
        customdata=np.c_[reg.beta_t, reg.beta_p],
        hovertemplate="%{y}: beta %{x:.2f}<br>t = %{customdata[0]:.2f}, p-value %{customdata[1]:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(zeroline=True, zerolinecolor="#bdbcb6", title="Esposizione (beta)")
    return _layout(fig, f"Esposizione ai fattori ({reg.modello})", 380)


def grafico_scomposizione_rendimento(a: Analisi) -> go.Figure:
    s = a.rendimento_fattori
    etichette = [_nome(k) if k != "Alfa" else "Alfa" for k in s.index] + ["Rendimento medio del fondo"]
    fig = go.Figure(go.Waterfall(
        x=etichette, y=list(s.values) + [s.sum()], measure=["relative"] * len(s) + ["total"],
        increasing={"marker": {"color": ACQUA}}, decreasing={"marker": {"color": ROSSO}},
        totals={"marker": {"color": BLU_SCURO}}, connector={"line": {"color": GRIGLIA}},
        text=[f"{v:+.2%}".replace(".", ",") for v in s.values] + [f"{s.sum():.2%}".replace(".", ",")],
        textposition="outside", hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "Da dove viene il rendimento medio annuo", 420)


def grafico_rischio(a: Analisi) -> go.Figure:
    s = a.rischio_fattori
    etichette = [_nome(k) if k != "Specifico" else "Rischio specifico" for k in s.index]
    colori = [COLORI_FATTORI.get(k, GRIGIO) for k in s.index]
    fig = go.Figure(go.Bar(x=etichette, y=s.values, marker_color=colori,
                           text=[f"{v:.0%}" for v in s.values], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%} della varianza<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="Quota della varianza")
    return _layout(fig, "Da dove viene il rischio", 380)


def grafico_mobili(a: Analisi) -> go.Figure:
    m = a.mobili
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    for f in [c for c in m.columns if c != "Alfa"]:
        fig.add_trace(go.Scatter(x=m.index, y=m[f], name=_nome(f), line={"color": COLORI_FATTORI.get(f, BLU),
                                                                          "width": 1.8},
                                 hovertemplate=f"{_nome(f)}: %{{y:.2f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=m.index, y=m["Alfa"], name="Alfa annuo", line={"color": TESTO, "width": 1.5},
                             fill="tozeroy", fillcolor="rgba(140,139,134,0.15)",
                             hovertemplate="Alfa %{y:.2%}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title="Beta", row=1, col=1)
    fig.update_yaxes(title="Alfa", tickformat=".0%", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Esposizioni stimate su finestre mobili di 36 mesi", 560)


def grafico_bootstrap(a: Analisi) -> go.Figure:
    t_sim = a.bootstrap["t simulati"]
    t = a.bootstrap["t osservato"]
    fig = go.Figure(go.Histogram(x=t_sim, nbinsx=60, marker_color="rgba(42,120,214,0.55)",
                                 name="Gestori senza capacità (simulati)",
                                 hovertemplate="t tra %{x}: %{y} simulazioni<extra></extra>"))
    fig.add_vline(x=t, line={"color": ROSSO if t < 0 else ACQUA, "width": 3},
                  annotation_text=f"Fondo: t = {t:.2f}".replace(".", ","), annotation_position="top")
    fig.update_xaxes(title="Statistica t dell'alfa")
    fig.update_yaxes(title="Simulazioni")
    return _layout(fig, "Talento o fortuna: il fondo contro 5.000 gestori senza capacità", 380)


def grafico_stile(a: Analisi) -> go.Figure:
    p = a.stile.pesi
    p = p[p > 0.005].sort_values()
    fig = go.Figure(go.Bar(y=p.index, x=p.values, orientation="h", marker_color=BLU_SCURO,
                           text=[f"{v:.0%}" for v in p.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, min(1.0, p.max() * 1.25)])
    return _layout(fig, "Stile effettivo: il portafoglio di indici che replica il fondo", 340)


def grafico_stile_mobile(a: Analisi) -> go.Figure:
    m = a.stile_mobile
    m = m.loc[:, m.max() > 0.02]
    fig = go.Figure()
    for i, c in enumerate(m.columns):
        fig.add_trace(go.Scatter(x=m.index, y=m[c], name=c, stackgroup="uno", mode="none",
                                 fillcolor=COLORI[i % len(COLORI)],
                                 hovertemplate=f"{c}: %{{y:.0%}}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Stile nel tempo (finestre mobili di 36 mesi)", 440)


def grafico_brinson(a: Analisi) -> go.Figure:
    p = a.attribuzione.per_segmento.sort_values("Totale")
    fig = go.Figure()
    for e in brinson.EFFETTI:
        fig.add_trace(go.Bar(y=p.index, x=p[e], name=e, orientation="h", marker_color=COLORI_EFFETTI[e],
                             hovertemplate=f"%{{y}}<br>{e}: %{{x:.2%}}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(tickformat=".1%", zeroline=True, zerolinecolor="#bdbcb6", title="Contributo al rendimento attivo")
    return _layout(fig, "Attribuzione per settore", 480)


def grafico_brinson_tempo(a: Analisi) -> go.Figure:
    c = a.attribuzione.cumulato
    fig = go.Figure()
    for e in brinson.EFFETTI:
        fig.add_trace(go.Scatter(x=c.index, y=c[e], name=e, line={"color": COLORI_EFFETTI[e], "width": 1.8},
                                 hovertemplate=f"{e}: %{{y:.2%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=c.index, y=c["Rendimento attivo"], name="Rendimento attivo (lordo)",
                             line={"color": TESTO, "width": 2.5}, hovertemplate="Attivo: %{y:.2%}<extra></extra>"))
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Effetti cumulati nel tempo (collegamento di Cariño)", 400)


def grafico_confronto(a: Analisi) -> go.Figure:
    c = a.confronto.sort_values("Alfa multifattore")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=c.index, x=c["Alfa CAPM"], name="Alfa CAPM (solo mercato)", orientation="h",
                         marker_color="rgba(140,139,134,0.6)", hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.add_trace(go.Bar(y=c.index, x=c["Alfa multifattore"], name=f"Alfa {a.regressione.modello}",
                         orientation="h", marker_color=[ACQUA if v > 0 else ROSSO for v in c["Alfa multifattore"]],
                         hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".1%", zeroline=True, zerolinecolor="#bdbcb6", title="Alfa annuo")
    return _layout(fig, "Alfa apparente e alfa vero dei fondi a confronto", 120 + 70 * len(c))


def _disponibili(a: Analisi) -> dict:
    g = {"crescita": grafico_crescita, "modelli": grafico_modelli, "esposizioni": grafico_esposizioni,
         "rendimento": grafico_scomposizione_rendimento, "rischio": grafico_rischio, "bootstrap": grafico_bootstrap}
    if not a.mobili.empty:
        g["mobili"] = grafico_mobili
    if a.stile is not None:
        g["stile"] = grafico_stile
    if a.stile_mobile is not None:
        g["stile_mobile"] = grafico_stile_mobile
    if a.attribuzione is not None:
        g["brinson"] = grafico_brinson
        g["brinson_tempo"] = grafico_brinson_tempo
    if a.confronto is not None:
        g["confronto"] = grafico_confronto
    return g


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede ``kaleido`` e un browser Chrome/Chromium)."""
    out = Path(cartella)
    out.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, funzione in _disponibili(a).items():
        percorso = out / f"{nome}.png"
        funzione(a).write_image(percorso, width=900, scale=scala)
        percorsi.append(percorso)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x: float, decimali: int = 1, segno: bool = False) -> str:
    if x is None or pd.isna(x):
        return "–"
    testo = f"{x:+.{decimali}%}" if segno else f"{x:.{decimali}%}"
    return testo.replace(".", ",").replace("-", "−")


def _num(x: float, decimali: int = 2) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{x:.{decimali}f}".replace(".", ",").replace("-", "−")


def _esposizioni_rilevanti(reg: fattori.Regressione) -> list[str]:
    """Fattori diversi dal mercato con esposizione significativa e non trascurabile, in ordine di importanza."""
    altri = [f for f in reg.fattori if f != "MKT" and reg.beta_p[f] < 0.05 and abs(reg.beta[f]) >= 0.1]
    return sorted(altri, key=lambda f: -abs(reg.beta[f]))


def _giudizio(a: Analisi) -> tuple[str, str]:
    """Una frase di sintesi e una classe CSS per il riquadro di esito."""
    reg = a.regressione
    capm = a.modelli.loc["CAPM"] if "CAPM" in a.modelli.index else None
    p_boot = a.bootstrap["p-value una coda"]
    rilevanti = _esposizioni_rilevanti(reg)
    stili = ", ".join(f"{_nome(f).lower()} ({_num(reg.beta[f])})" for f in rilevanti)

    frasi = []
    if a.metriche is not None:
        attivo = a.metriche["Rendimento attivo"]
        frasi.append(f"Al netto dei costi il fondo ha reso {_pct(abs(attivo))} l'anno "
                     f"{'più' if attivo >= 0 else 'meno'} del benchmark.")
    if capm is not None and capm["Alfa annuo"] > 0 and reg.alfa < capm["Alfa annuo"] - 0.005:
        frasi.append(f"Rispetto al solo mercato sembra creare un alfa del {_pct(capm['Alfa annuo'])} annuo, "
                     f"ma tenendo conto dei fattori di stile l'alfa diventa {_pct(reg.alfa, segno=True)}.")
    else:
        frasi.append(f"L'alfa stimato con il modello {reg.modello} è {_pct(reg.alfa, segno=True)} annuo.")
    if stili:
        frasi.append(f"Il rendimento si spiega con esposizioni a fattori replicabili a basso costo: {stili}.")

    if reg.alfa > 0 and reg.alfa_p < 0.05 and p_boot < 0.05:
        esito = "ok"
        frasi.append("L'alfa è statisticamente significativo e difficilmente attribuibile alla fortuna: "
                     "c'è evidenza di capacità del gestore.")
    elif reg.alfa < 0 and reg.alfa_p < 0.05:
        esito = "ko"
        frasi.append("L'alfa negativo è significativo: il fondo ha restituito meno di quanto le sue esposizioni "
                     "avrebbero reso con strumenti passivi.")
    else:
        esito = "attenzione"
        frasi.append(f"Non c'è evidenza statistica di capacità del gestore (p-value {_num(reg.alfa_p, 2)}): "
                     "la differenza dal mercato è compatibile con il caso e con le esposizioni ai fattori.")
    if a.ter:
        frasi.append(f"Il costo annuo del fondo è {_pct(a.ter, 2)}.")
    return " ".join(frasi), esito


def build_report(
    a: Analisi,
    output: str | Path,
    client_name: str = "Cliente",
    data_note: str = "",
    offline: bool = False,
) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    sintesi, esito = _giudizio(a)
    config = {"displaylogo": False, "responsive": True}
    grafici = {nome: f(a).to_html(full_html=False, include_plotlyjs=False, config=config)
               for nome, f in _disponibili(a).items()}
    reg = a.regressione
    r = a.rendimenti
    m = a.metriche or {}

    kpi = [("Rendimento annuo netto", _pct(metriche.rendimento_annuo(r)))]
    if m:
        kpi += [("Rispetto al benchmark", _pct(m["Rendimento attivo"], segno=True)),
                ("Tracking error", _pct(m["Tracking error"]))]
    kpi += [
        ("Alfa CAPM", _pct(a.modelli.loc["CAPM", "Alfa annuo"], segno=True) if "CAPM" in a.modelli.index else "–"),
        ("Alfa multifattore", _pct(reg.alfa, segno=True)),
        ("Spiegato dai fattori (R²)", _pct(reg.r2, 0)),
    ]

    modelli = [{"nome": n, "alfa": _pct(x["Alfa annuo"], 2, True), "t": _num(x["t alfa"]),
                "p": _num(x["p-value alfa"], 3), "beta": _num(x["Beta mercato"]), "r2": _num(x["R² corretto"], 3),
                "spec": _pct(x["Rischio specifico"]), "bic": _num(x["BIC"], 0), "scelto": n == reg.modello}
               for n, x in a.modelli.iterrows()]
    migliore_bic = a.modelli["BIC"].idxmin()
    esposizioni = [{"nome": _nome(f), "codice": f, "beta": _num(reg.beta[f]), "t": _num(reg.beta_t[f]),
                    "p": _num(reg.beta_p[f], 3), "contributo": _pct(a.rendimento_fattori[f], 2, True),
                    "rischio": _pct(a.rischio_fattori[f], 0), "sig": reg.beta_p[f] < 0.05}
                   for f in reg.fattori]

    metr = []
    if m:
        etichette = [("Rendimento annuo", "pct"), ("Volatilità", "pct"), ("Sharpe", "num"),
                     ("Massima perdita", "pct"), ("Tracking error", "pct"), ("Information ratio", "num"),
                     ("Beta", "num"), ("Up capture", "pct0"), ("Down capture", "pct0"),
                     ("Mesi sopra il benchmark", "pct0")]
        mb = metriche.sintesi(a.benchmark.reindex(r.index), a.benchmark.reindex(r.index), a.fattori["RF"])
        fmt = {"pct": _pct, "num": _num, "pct0": lambda x: _pct(x, 0)}
        metr = [{"nome": e, "fondo": fmt[t](m[e]), "bench": fmt[t](mb[e]) if e not in (
            "Tracking error", "Information ratio") else "–"} for e, t in etichette]

    stile_ctx = None
    if a.stile is not None:
        p = a.stile.pesi[a.stile.pesi > 0.005].sort_values(ascending=False)
        stile_ctx = {"r2": _pct(a.stile.r2, 0), "selezione": _pct(a.stile.selezione, 2, True),
                     "te": _pct(a.stile.tracking_error),
                     "pesi": [{"nome": k, "peso": _pct(v, 0)} for k, v in p.items()],
                     "principale": p.index[0], "quota": _pct(p.iloc[0], 0),
                     "mobile": a.stile_mobile is not None}

    brinson_ctx = None
    if a.attribuzione is not None:
        t = a.attribuzione.totale
        settori = [{"nome": s, "pf": _pct(x["Peso medio fondo"]), "pb": _pct(x["Peso medio benchmark"]),
                    "rf": _pct(x["Rendimento fondo"]), "rb": _pct(x["Rendimento benchmark"]),
                    "all": _pct(x["Allocazione"], 2, True), "sel": _pct(x["Selezione"], 2, True),
                    "int": _pct(x["Interazione"], 2, True), "tot": _pct(x["Totale"], 2, True),
                    "neg": x["Totale"] < 0}
                   for s, x in a.attribuzione.per_segmento.iterrows()]
        principale = max(brinson.EFFETTI, key=lambda e: abs(t[e]))
        costi = None
        if a.ter:
            costi = _pct(a.ter * a.mesi_attribuzione / 12, 2)
        brinson_ctx = {"mesi": a.mesi_attribuzione, "fondo": _pct(t["Rendimento fondo"]),
                       "bench": _pct(t["Rendimento benchmark"]), "attivo": _pct(t["Rendimento attivo"], 2, True),
                       "all": _pct(t["Allocazione"], 2, True), "sel": _pct(t["Selezione"], 2, True),
                       "int": _pct(t["Interazione"], 2, True), "principale": principale.lower(),
                       "settori": settori, "active_share": _pct(a.active_share, 0), "costi": costi,
                       "selezione_positiva": t["Selezione"] > 0,
                       "migliore": a.attribuzione.per_segmento["Totale"].idxmax(),
                       "peggiore": a.attribuzione.per_segmento["Totale"].idxmin()}

    confronto = []
    if a.confronto is not None:
        for n, x in a.confronto.sort_values("Alfa multifattore", ascending=False).iterrows():
            confronto.append({"nome": n, "rend": _pct(x.get("Rendimento annuo")), "vol": _pct(x.get("Volatilità")),
                              "te": _pct(x.get("Tracking error")), "ir": _num(x.get("Information ratio")),
                              "capm": _pct(x["Alfa CAPM"], 2, True), "multi": _pct(x["Alfa multifattore"], 2, True),
                              "t": _num(x["t multifattore"]), "r2": _pct(x["R²"], 0), "questo": n == a.fondo,
                              "sig": abs(x["t multifattore"]) >= 2})

    boot = a.bootstrap
    contesto = {
        "client": client_name,
        "fondo": a.fondo,
        "benchmark": a.nome_benchmark,
        "today": date.today().strftime("%d/%m/%Y"),
        "periodo": f"{r.index[0]:%m/%Y} al {r.index[-1]:%m/%Y}",
        "mesi": len(r),
        "sintesi": sintesi,
        "esito": esito,
        "kpi": kpi,
        "grafici": grafici,
        "metriche": metr,
        "modelli": modelli,
        "modello": reg.modello,
        "migliore_bic": migliore_bic,
        "esposizioni": esposizioni,
        "alfa": _pct(reg.alfa, 2, True),
        "alfa_p": _num(reg.alfa_p, 3),
        "alfa_t": _num(reg.alfa_t),
        "appraisal": _num(reg.appraisal_ratio),
        "rf_medio": _pct(a.rendimento_fattori["RF"], 2),
        "rendimento_medio": _pct(a.rendimento_fattori.sum(), 2),
        "specifico": _pct(a.rischio_fattori["Specifico"], 0),
        "sistematico": _pct(1 - a.rischio_fattori["Specifico"], 0),
        "rilevanti": ", ".join(_nome(f).lower() for f in _esposizioni_rilevanti(reg)) or "nessuno",
        "boot_p": _num(boot["p-value una coda"], 3),
        "boot_soglia": _num(boot["t al 95% per fortuna"]),
        "boot_t": _num(boot["t osservato"]),
        "boot_n": f"{len(boot['t simulati']):,}".replace(",", "."),
        "mobili": not a.mobili.empty,
        "stile": stile_ctx,
        "brinson": brinson_ctx,
        "confronto": confronto,
        "ter": _pct(a.ter, 2) if a.ter else None,
        "data_note": data_note,
        "plotly_inline": get_plotlyjs() if offline else None,
        "plotly_cdn": PLOTLY_CDN,
    }
    env = Environment(loader=PackageLoader("factor_attribution", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(**contesto)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
