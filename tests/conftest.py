import pytest

from factor_attribution.dati import generate_sample_universe


@pytest.fixture(scope="session")
def universo():
    return generate_sample_universe()


@pytest.fixture(scope="session")
def fattori(universo):
    return universo["fattori"]


@pytest.fixture(scope="session")
def fondi(universo):
    return universo["fondi"]
