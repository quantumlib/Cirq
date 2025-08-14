# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq


def test_version() -> None:
    assert cirq.__version__ == "1.6.1"
