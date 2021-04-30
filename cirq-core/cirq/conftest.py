import inspect
import os
import matplotlib.pyplot as plt


def pytest_configure(config):
    # Use matplotlib agg backend which does not require a display.
    plt.switch_backend('agg')
    os.environ['CIRQ_TESTING'] = "true"


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem._obj):
        # coverage: ignore
        raise ValueError(
            f'{pyfuncitem._obj.__name__} is async but not '
            f'decorated with "@pytest.mark.asyncio".'
        )
