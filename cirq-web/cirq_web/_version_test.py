# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq_web


def test_version() -> None:
    assert cirq_web.__version__ == "1.8.0.dev0"
