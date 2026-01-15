import pytest


def pytest_addoption(parser):
    parser.addoption("--generate-baseline", default=False, action="store_true")


@pytest.fixture
def generate_baseline(request):
    return request.config.getoption("--generate-baseline")
