# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq


def test_version() -> None:
    assert cirq.__version__ == "1.7.0.dev0"
