import numpy as np
import pandas as pd
import pytest

from factor_attribution import fattori as F
from factor_attribution.dati import FATTORI


def _fondo_sintetico(fattori, beta, alfa_mensile, rumore=0.004, seed=1):
    rng = np.random.default_rng(seed)
    b = pd.Series(beta).reindex(FATTORI).fillna(0.0)
    r = fattori["RF"] + alfa_mensile + fattori[FATTORI] @ b + rng.normal(0, rumore, len(fattori))
    return r.rename("Test")


def test_regressione_ritrova_esposizioni_e_alfa(fattori):
    beta = {"MKT": 0.9, "SMB": 0.3, "HML": -0.4, "RMW": 0.2, "MOM": 0.1}
    r = _fondo_sintetico(fattori, beta, 0.002, rumore=0.0005)
    reg = F.regressione(r, fattori)
    for k, v in beta.items():
        assert reg.beta[k] == pytest.approx(v, abs=0.02)
    assert reg.alfa == pytest.approx(0.024, abs=0.002)
    assert reg.r2 > 0.99
    assert reg.alfa_t > 10


def test_alfa_apparente_sparisce_con_i_fattori(fattori):
    # Fondo senza capacità ma con forte esposizione value: alfa CAPM positivo, multifattore nullo
    r = _fondo_sintetico(fattori, {"MKT": 1.0, "HML": 0.8}, 0.0)
    m = F.confronta_modelli(r, fattori)
    assert m.loc["CAPM", "Alfa annuo"] > 0.015
    assert abs(m.loc["Fama-French 3", "Alfa annuo"]) < 0.01
    assert m.loc["Fama-French 3", "R² corretto"] > m.loc["CAPM", "R² corretto"]


def test_scomposizione_rendimento_somma_al_rendimento_medio(fondi, fattori):
    r = fondi["Fondo Europa Value Attivo"]
    s = F.scomposizione_rendimento(F.regressione(r, fattori))
    assert s.sum() == pytest.approx(r.mean() * 12, abs=1e-10)


def test_scomposizione_rischio_somma_a_uno(fondi, fattori):
    s = F.scomposizione_rischio(F.regressione(fondi["Fondo Europa Qualità"], fattori))
    assert s.sum() == pytest.approx(1.0, abs=1e-10)
    assert 0 < s["Specifico"] < 1


def test_modello_personalizzato_e_fattori_mancanti(fondi, fattori):
    reg = F.regressione(fondi["ETF Azionario Europa"], fattori, ["MKT", "HML"])
    assert reg.fattori == ["MKT", "HML"] and reg.modello == "MKT + HML"
    with pytest.raises(ValueError, match="mancanti"):
        F.regressione(fondi["ETF Azionario Europa"], fattori[["MKT", "RF"]], "Fama-French 3")


def test_pochi_mesi(fattori):
    with pytest.raises(ValueError, match="24 mesi"):
        F.regressione(_fondo_sintetico(fattori, {"MKT": 1}, 0).iloc[:12], fattori, "CAPM")


def test_esposizioni_mobili(fondi, fattori):
    m = F.esposizioni_mobili(fondi["Fondo Europa Value Attivo"], fattori, finestra=36)
    assert len(m) == len(fondi) - 35
    assert list(m.columns) == ["Alfa", *F.MODELLI[F.MODELLO_COMPLETO]]
    assert m["HML"].mean() > 0.4


def test_bootstrap_distingue_talento_da_fortuna(fattori):
    bravo = F.regressione(_fondo_sintetico(fattori, {"MKT": 1.0}, 0.004, rumore=0.01), fattori, "CAPM")
    nullo = F.regressione(_fondo_sintetico(fattori, {"MKT": 1.0}, 0.0, rumore=0.01, seed=3), fattori, "CAPM")
    b1 = F.bootstrap_alfa(bravo, n=1000)
    b2 = F.bootstrap_alfa(nullo, n=1000)
    assert b1["p-value una coda"] < 0.01
    assert b2["p-value una coda"] > 0.05
    assert 1.3 < b1["t al 95% per fortuna"] < 2.2
    assert len(b1["t simulati"]) == 1000


def test_lag_newey_west():
    assert F.lag_newey_west(100) == 4
    assert F.lag_newey_west(192) == 4
