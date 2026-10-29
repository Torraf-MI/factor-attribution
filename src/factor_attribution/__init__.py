"""Modelli fattoriali, style analysis e attribuzione di performance per valutare fondi e gestori."""

from .brinson import Attribuzione, active_share, attribuzione_brinson
from .dati import (
    generate_sample_factors,
    generate_sample_universe,
    load_csv,
    load_fattori,
    parse_french_csv,
    rendimenti_mensili,
    scarica_fattori_french,
)
from .fattori import (
    MODELLI,
    Regressione,
    bootstrap_alfa,
    confronta_modelli,
    esposizioni_mobili,
    regressione,
    scomposizione_rendimento,
    scomposizione_rischio,
)
from .metriche import information_ratio, sintesi, tracking_error
from .stile import Stile, analisi_stile, stile_mobile

__all__ = [
    "MODELLI",
    "Attribuzione",
    "Regressione",
    "Stile",
    "active_share",
    "analisi_stile",
    "bootstrap_alfa",
    "attribuzione_brinson",
    "confronta_modelli",
    "esposizioni_mobili",
    "generate_sample_factors",
    "generate_sample_universe",
    "information_ratio",
    "load_csv",
    "load_fattori",
    "parse_french_csv",
    "regressione",
    "rendimenti_mensili",
    "scarica_fattori_french",
    "scomposizione_rendimento",
    "scomposizione_rischio",
    "sintesi",
    "stile_mobile",
    "tracking_error",
]

__version__ = "0.1.0"
