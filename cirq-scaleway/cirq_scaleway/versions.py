import importlib.metadata
import platform

from typing import Final

CIRQ_VERSION: Final = importlib.metadata.version("cirq")
CIRQ_SCALEWAY_PROVIDER_VERSION: Final = importlib.metadata.version("cirq-scaleway")

__version__: Final = CIRQ_SCALEWAY_PROVIDER_VERSION

USER_AGENT: Final = " ".join(
    [
        f"cirq-scaleway/{CIRQ_SCALEWAY_PROVIDER_VERSION}",
        f"({platform.system()}; {platform.python_implementation()}/{platform.python_version()})",
        f"cirq/{CIRQ_VERSION}",
    ]
)
