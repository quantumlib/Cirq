# pylint: disable=wrong-or-nonexistent-copyright-notice
import pack2._version  # type: ignore[import-not-found]


def test_version() -> None:
    assert pack2._version.__version__ == "1.2.3.dev"
