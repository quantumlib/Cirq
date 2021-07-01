import os


def pytest_configure(config):
    os.environ['CIRQ_TESTING'] = "true"
