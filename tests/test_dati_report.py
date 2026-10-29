import numpy as np
import pandas as pd
import pytest

from factor_attribution import cli, dati, metriche
from factor_attribution.report import analizza, build_report

FRENCH = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202412 CRSP database.
The 1-month TBill return is from Ibbotson and Associates Inc.

,Mkt-RF,SMB,HML,RMW,CMA,RF
202301,   7.12,   1.10,  -2.10,   0.40,  -1.20,   0.35
202302,  -2.50,   0.30,   1.20,  -0.80,   0.50,   0.34
202303,   1.80,  -2.00,  -3.10,   1.30,  -0.70,   0.36

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
 2023,  20.10,  -1.00,  -5.00,   3.00,  -2.00,   5.00
"""


def test_parse_french():
    f = dati.parse_french_csv(FRENCH)
    assert list(f.columns) == ["MKT", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(f) == 3  # solo la tabella mensile
    assert f.index[0] == pd.Timestamp("2023-01-31")
    assert f.loc["2023-01-31", "MKT"] == pytest.approx(0.0712)


def test_fattori_dimostrativi(fattori):
    for k, (media, vol) in dati.PARAMETRI_FATTORI.items():
        assert fattori[k].mean() == pytest.approx(media, abs=1e-6)
        assert fattori[k].std() == pytest.approx(vol, abs=1e-6)
    assert fattori.index[-1] == pd.Timestamp("2025-12-31")


def test_load_fattori_percentuali(tmp_path, fattori):
    path = tmp_path / "f.csv"
    (fattori * 100).rename(columns={"MKT": "Mkt-RF"}).to_csv(path)
    f = dati.load_fattori(path)
    assert np.allclose(f["MKT"], fattori["MKT"])


def test_rendimenti_mensili_da_prezzi_giornalieri():
    giorni = pd.bdate_range("2024-01-01", "2024-04-30")
    prezzi = pd.DataFrame({"A": np.linspace(100, 110, len(giorni))}, index=giorni)
    r = dati.rendimenti_mensili(prezzi)
    assert len(r) == 3 and r.index[0] == pd.Timestamp("2024-02-29")
    assert (1 + r["A"]).prod() == pytest.approx(110 / prezzi.resample("ME").last()["A"].iloc[0])
    gia_rendimenti = dati.rendimenti_mensili(r)
    assert np.allclose(gia_rendimenti["A"], r["A"])


def test_metriche_benchmark(fondi):
    b = fondi["Indice Azionario Europa"]
    m = metriche.sintesi(fondi["ETF Azionario Europa"], b)
    assert m["Tracking error"] < 0.01
    assert m["Beta"] == pytest.approx(1.0, abs=0.02)
    assert np.isnan(metriche.information_ratio(b, b))


def test_report_completo(tmp_path, universo):
    u = universo
    bench = "Indice Azionario Europa"
    a = analizza(u["fondi"]["Fondo Europa Value Attivo"], u["fattori"], benchmark=u["fondi"][bench],
                 indici=u["indici"], settori=u["settori"], altri_fondi=u["fondi"].drop(columns=bench),
                 ter=0.0165, n_bootstrap=300)
    path = build_report(a, tmp_path / "r.html", client_name="Test")
    html = path.read_text(encoding="utf-8")
    for testo in ["Talento o fortuna", "Analisi di stile", "Attribuzione di performance", "Confronto con altri fondi",
                  "Fondo Europa Qualità"]:
        assert testo in html
    assert a.mesi_attribuzione == 36


def test_cli_demo(tmp_path, capsys):
    out = tmp_path / "report.html"
    assert cli.main(["--demo", "--output", str(out), "--bootstrap", "200"]) == 0
    assert out.exists()
    assert "Brinson" in capsys.readouterr().out


def test_cli_da_csv(tmp_path, universo):
    fondi = tmp_path / "fondi.csv"
    fattori = tmp_path / "fattori.csv"
    u = universo
    # Valori quota invece di rendimenti: vengono convertiti in rendimenti mensili
    (100 * (1 + u["fondi"]).cumprod()).to_csv(fondi)
    u["fattori"].to_csv(fattori)
    out = tmp_path / "r.html"
    assert cli.main(["--fondi", str(fondi), "--fattori", str(fattori), "--fondo", "Fondo Europa Qualità",
                     "--benchmark", "Indice Azionario Europa", "--output", str(out), "--bootstrap", "200"]) == 0
    assert "Fondo Europa Qualità" in out.read_text(encoding="utf-8")
