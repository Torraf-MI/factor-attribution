"""Riga di comando: analisi fattoriale, di stile e attribuzione di performance di un fondo.

Esempi::

    factor-attribution --demo --cliente "Mario Rossi"
    factor-attribution --demo --fondo "Fondo Europa Qualità"
    factor-attribution --fondi data/fondi_esempio.csv --fattori data/fattori_esempio.csv \
        --fondo "Fondo Europa Value Attivo" --benchmark "Indice Azionario Europa" --settori data/settori_esempio.csv
    factor-attribution --fondi valori_quota.csv --french Europe --fondo "Il mio fondo"
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import dati
from .report import _num, _pct, analizza, build_report, salva_grafici


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="factor-attribution", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fondi", help="CSV con una colonna data e una colonna per fondo (prezzi o rendimenti)")
    src.add_argument("--demo", action="store_true", help="Usa l'universo dimostrativo sintetico (2010-2025)")
    fattori = p.add_mutually_exclusive_group()
    fattori.add_argument("--fattori", help="CSV di fattori mensili (MKT o Mkt-RF, SMB, HML, RMW, CMA, MOM, RF)")
    fattori.add_argument("--french", choices=["Europe", "Developed", "US"],
                         help="Scarica i fattori dalla Kenneth French Data Library")
    p.add_argument("--fondo", help="Fondo da analizzare (colonna del CSV; default: il primo)")
    p.add_argument("--benchmark", help="Colonna del CSV dei fondi da usare come benchmark")
    p.add_argument("--indici", help="CSV di indici di stile (prezzi o rendimenti) per la style analysis")
    p.add_argument("--settori", help="CSV per l'attribuzione di Brinson: data, settore, peso_fondo, "
                                     "peso_benchmark, rend_fondo, rend_benchmark")
    p.add_argument("--mesi-attribuzione", type=int, default=36,
                   help="Mesi più recenti per l'attribuzione di Brinson (0 = tutto il periodo)")
    p.add_argument("--modello", default="Fama-French 5 + Momentum",
                   choices=["CAPM", "Fama-French 3", "Carhart 4", "Fama-French 5", "Fama-French 5 + Momentum"])
    p.add_argument("--finestra", type=int, default=36, help="Mesi delle stime mobili (default 36)")
    p.add_argument("--ter", type=float, help="Costo annuo del fondo, ad esempio 0.015 per l'1,5%%")
    p.add_argument("--bootstrap", type=int, default=5000, help="Simulazioni per il test talento o fortuna")
    p.add_argument("--senza-confronto", action="store_true", help="Non confrontare con gli altri fondi del CSV")
    p.add_argument("--cliente", default="Cliente", help="Nome del cliente nel report")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    note, indici, settori, ter = "", None, None, args.ter
    if args.demo:
        u = dati.generate_sample_universe()
        fondi, fattori, indici, settori = u["fondi"], u["fattori"], u["indici"], u["settori"]
        nome = args.fondo or dati.FONDO_ESEMPIO
        nome_bench = args.benchmark or dati.BENCHMARK_ESEMPIO
        ter = ter if ter is not None else dati.TER_ESEMPIO.get(nome)
        if nome != dati.FONDO_ESEMPIO:
            settori = None  # i pesi per settore dimostrativi sono solo quelli del fondo di esempio
        note = "Analisi svolta su dati sintetici dimostrativi, non su fondi o fattori reali."
    else:
        fondi = dati.rendimenti_mensili(dati.load_csv(args.fondi))
        if args.fattori:
            fattori = dati.load_fattori(args.fattori)
        elif args.french:
            fattori = dati.scarica_fattori_french(args.french)
        else:
            raise SystemExit("Indica i fattori con --fattori file.csv oppure --french Europe")
        nome = args.fondo or fondi.columns[0]
        nome_bench = args.benchmark
        if args.indici:
            indici = dati.rendimenti_mensili(dati.load_csv(args.indici))
        if args.settori:
            settori = pd.read_csv(args.settori)

    if nome not in fondi:
        raise SystemExit(f"Fondo {nome!r} non trovato tra: {', '.join(fondi.columns)}")
    if nome_bench and nome_bench not in fondi:
        raise SystemExit(f"Benchmark {nome_bench!r} non trovato tra: {', '.join(fondi.columns)}")
    benchmark = fondi[nome_bench].dropna() if nome_bench else None
    altri = None
    if not args.senza_confronto:
        altri = fondi.drop(columns=[c for c in [nome_bench] if c])
        altri = altri if altri.shape[1] > 1 else None

    a = analizza(fondi[nome].dropna(), fattori, benchmark=benchmark, indici=indici, settori=settori,
                 altri_fondi=altri, ter=ter, modello=args.modello, finestra=args.finestra,
                 mesi_attribuzione=args.mesi_attribuzione or None, n_bootstrap=args.bootstrap)
    path = build_report(a, args.output, client_name=args.cliente, data_note=note, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    pd.set_option("display.width", 160)
    print(f"{a.fondo}: {len(a.rendimenti)} mesi dal {a.rendimenti.index[0]:%m/%Y} al {a.rendimenti.index[-1]:%m/%Y}")
    print("\nAlfa per modello")
    print(a.modelli[["Alfa annuo", "t alfa", "Beta mercato", "R² corretto"]].to_string(
        formatters={"Alfa annuo": "{:+.2%}".format, "t alfa": "{:.2f}".format, "Beta mercato": "{:.2f}".format,
                    "R² corretto": "{:.3f}".format}))
    reg = a.regressione
    print(f"\nEsposizioni ({reg.modello})")
    print(pd.DataFrame({"Beta": reg.beta, "t": reg.beta_t}).to_string(float_format="{:.2f}".format))
    print(f"\nTalento o fortuna: p-value del bootstrap {_num(a.bootstrap['p-value una coda'], 3)}")
    if a.stile is not None:
        p = a.stile.pesi[a.stile.pesi > 0.005].sort_values(ascending=False)
        print(f"\nStile (R² {_pct(a.stile.r2, 0)}): " + ", ".join(f"{k} {_pct(v, 0)}" for k, v in p.items()))
    if a.attribuzione is not None:
        t = a.attribuzione.totale
        print(f"\nBrinson, ultimi {a.mesi_attribuzione} mesi: attivo lordo {_pct(t['Rendimento attivo'], 2)} = "
              f"allocazione {_pct(t['Allocazione'], 2)} + selezione {_pct(t['Selezione'], 2)} + "
              f"interazione {_pct(t['Interazione'], 2)}")
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
