"""Uso del progetto come libreria: alfa vero, talento o fortuna, stile e attribuzione di Brinson."""

from factor_attribution import (
    analisi_stile,
    attribuzione_brinson,
    bootstrap_alfa,
    confronta_modelli,
    generate_sample_universe,
    regressione,
    scomposizione_rendimento,
    stile_mobile,
)
from factor_attribution.dati import fondo_bilanciato_esempio

u = generate_sample_universe()  # oppure load_fattori("fattori.csv") / scarica_fattori_french("Europe")
fattori, fondi, indici, settori = u["fattori"], u["fondi"], u["indici"], u["settori"]
fondo = fondi["Fondo Europa Value Attivo"]

# 1. L'alfa cambia con il modello: rispetto al solo mercato il fondo sembra bravo, con i fattori no
print(confronta_modelli(fondo, fattori)[["Alfa annuo", "t alfa", "R² corretto"]].round(3))

# 2. Esposizioni e da dove viene il rendimento medio annuo
reg = regressione(fondo, fattori, "Fama-French 5 + Momentum")
print(reg.beta.round(2))
print(scomposizione_rendimento(reg).round(4))

# 3. Talento o fortuna: bootstrap dell'alfa sotto l'ipotesi di gestore senza capacità
b = bootstrap_alfa(reg, n=5000)
print(f"t dell'alfa {b['t osservato']:.2f}, p-value {b['p-value una coda']:.3f}")

# 4. Style analysis di Sharpe: il portafoglio di indici che replica il fondo
st = analisi_stile(fondo, indici)
print(st.pesi[st.pesi > 0.01].round(2), f"R² {st.r2:.1%}")

# 5. Deriva di stile di un bilanciato flessibile, dai soli valori quota
bilanciato = fondo_bilanciato_esempio(indici)
m = stile_mobile(bilanciato, indici, finestra=36)
azioni = m[[c for c in m if c.startswith("Azioni")]].sum(axis=1)
inizio, fine = azioni.index[0], azioni.index[-1]
print(f"Azionario stimato: {azioni.iloc[0]:.0%} nel {inizio:%Y}, {azioni.iloc[-1]:.0%} nel {fine:%Y}")

# 6. Attribuzione di Brinson sugli ultimi 36 mesi, collegata con Cariño
date = sorted(settori["data"].unique())[-36:]
a = attribuzione_brinson(settori[settori["data"].isin(date)])
print(a.totale.round(4))
print(a.per_segmento[["Allocazione", "Selezione", "Interazione", "Totale"]].round(4))
