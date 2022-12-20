# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq


def test_version():
    assert cirq.__version__ == "1.2.0.dev"
