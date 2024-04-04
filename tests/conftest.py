import pytest 

# ! This allows to run tests marked `external` only with the command line `pytest --external`

def pytest_addoption(parser):
    parser.addoption("--external", action="store_true",
                     help="run the tests only in case of that command line (marked with marker @external)")

def pytest_runtest_setup(item):
    if 'external' in item.keywords and not item.config.getoption("--external"):
        pytest.skip("need --external option to run this test")