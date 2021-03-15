import inspect

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config):
    # Use matplotlib agg backend which does not require a display.
    plt.switch_backend('agg')
    config.addinivalue_line("markers", "subprocess: mark tests to be run in separate processes")


def pytest_collection_modifyitems(config, items):
    keywordexpr = config.option.keyword
    markexpr = config.option.markexpr
    if keywordexpr or markexpr:
        return  # let pytest handle this

    skip_subprocess_marker = pytest.mark.skip(
        reason='subprocess marker requires running pytest --forked'
    )
    for item in items:
        if 'subprocess' in item.keywords:
            item.add_marker(skip_subprocess_marker)


def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem._obj):
        # coverage: ignore
        raise ValueError(
            f'{pyfuncitem._obj.__name__} is async but not '
            f'decorated with "@pytest.mark.asyncio".'
        )
