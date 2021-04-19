import inspect
import os
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_cirq_testing_envvar():
    os.environ['CIRQ_TESTING'] = "true"


def pytest_configure(config):
    # Use matplotlib agg backend which does not require a display.
    plt.switch_backend('agg')


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem._obj):
        # coverage: ignore
        raise ValueError(
            f'{pyfuncitem._obj.__name__} is async but not '
            f'decorated with "@pytest.mark.asyncio".'
        )
