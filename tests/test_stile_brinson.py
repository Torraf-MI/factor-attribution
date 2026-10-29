import numpy as np
import pandas as pd
import pytest

from factor_attribution import brinson as B
from factor_attribution import stile as S
from factor_attribution.dati import fondo_bilanciato_esempio


def test_stile_ritrova_i_pesi(universo):
    indici = universo["indici"]
    pesi = pd.Series({"Azioni Europa Value": 0.5, "Obbligazioni Gov. EUR": 0.3, "Monetario EUR": 0.2})
    rng = np.random.default_rng(0)
    r = indici[pesi.index] @ pesi + rng.normal(0, 0.0005, len(indici))
    st = S.analisi_stile(r, indici)
    assert st.pesi.sum() == pytest.approx(1.0)
    assert (st.pesi >= 0).all()
    for k, v in pesi.items():
        assert st.pesi[k] == pytest.approx(v, abs=0.03)
    assert st.r2 > 0.99


def test_stile_mobile_riconosce_la_deriva(universo):
    indici = universo["indici"]
    m = S.stile_mobile(fondo_bilanciato_esempio(indici), indici, 36)
    azioni = m[[c for c in m if c.startswith("Azioni")]].sum(axis=1)
    assert azioni.iloc[-1] > azioni.iloc[0] + 0.15  # l'azionario sale dal 30% al 60%
    assert m.iloc[0]["Azioni Europa Value"] > m.iloc[0]["Azioni Europa Growth"]
    assert m.iloc[-1]["Azioni Europa Growth"] > m.iloc[-1]["Azioni Europa Value"]


def _esempio_libro():
    # Esempio a un periodo con tre settori, calcolabile a mano
    return pd.DataFrame({
        "data": ["2024-01-31"] * 3,
        "settore": ["A", "B", "C"],
        "peso_fondo": [0.5, 0.3, 0.2],
        "peso_benchmark": [0.4, 0.4, 0.2],
        "rend_fondo": [0.05, 0.02, -0.01],
        "rend_benchmark": [0.04, 0.03, -0.02],
    })


def test_brinson_periodo_singolo_a_mano():
    a = B.attribuzione_brinson(_esempio_libro())
    rb = 0.4 * 0.04 + 0.4 * 0.03 + 0.2 * -0.02  # 0.024
    attesa_all_a = 0.1 * (0.04 - rb)
    assert a.per_segmento.loc["A", "Allocazione"] == pytest.approx(attesa_all_a)
    assert a.per_segmento.loc["B", "Selezione"] == pytest.approx(0.4 * (0.02 - 0.03))
    assert a.per_segmento.loc["A", "Interazione"] == pytest.approx(0.1 * 0.01)
    assert a.totale[B.EFFETTI].sum() == pytest.approx(a.totale["Rendimento attivo"])


def test_bhb_e_fachler_stesso_totale():
    d = B.verifica(_esempio_libro())
    f = B.effetti_periodo(d, "fachler")[B.EFFETTI].sum().sum()
    b = B.effetti_periodo(d, "bhb")[B.EFFETTI].sum().sum()
    assert f == pytest.approx(b)
    with pytest.raises(ValueError):
        B.effetti_periodo(d, "altro")


def test_collegamento_carino_esatto_su_piu_periodi(universo):
    a = B.attribuzione_brinson(universo["settori"])
    t = a.totale
    assert t[B.EFFETTI].sum() == pytest.approx(t["Rendimento attivo"], abs=1e-12)
    # Il cumulato all'ultima data coincide con il totale, e ogni anno somma al proprio rendimento attivo
    assert a.cumulato.iloc[-1][B.EFFETTI].sum() == pytest.approx(t["Rendimento attivo"], abs=1e-12)
    assert np.allclose(a.annuale[B.EFFETTI].sum(axis=1), a.annuale["Rendimento attivo"])


def test_rendimenti_coerenti_con_i_fondi(universo):
    # Il rendimento lordo dai settori meno i costi è il rendimento netto del fondo
    from factor_attribution.dati import FONDO_ESEMPIO, TER_ESEMPIO

    a = B.attribuzione_brinson(universo["settori"])
    netto = a.per_periodo["Rendimento fondo"] - TER_ESEMPIO[FONDO_ESEMPIO] / 12
    assert np.allclose(netto.to_numpy(), universo["fondi"][FONDO_ESEMPIO].to_numpy(), atol=2e-6)


def test_verifica_pesi():
    d = _esempio_libro()
    d.loc[0, "peso_fondo"] = 0.6
    with pytest.raises(ValueError, match="non sommano a 1"):
        B.attribuzione_brinson(d)
    with pytest.raises(ValueError, match="mancanti"):
        B.attribuzione_brinson(d.drop(columns="rend_fondo"))


def test_active_share():
    assert B.active_share(pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 0.5, "B": 0.5})) == 0
    assert B.active_share(pd.Series({"A": 1.0}), pd.Series({"B": 1.0})) == pytest.approx(1.0)
