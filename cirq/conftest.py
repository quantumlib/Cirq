import matplotlib.pyplot as plt


def pytest_configure(config):
    # Use matplotlib agg backend which does not require a display.
    plt.switch_backend('agg')
