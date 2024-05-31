# pylint: disable=wrong-or-nonexistent-copyright-notice
import pack2._version  # type: ignore


def test_version():
    assert pack2._version.__version__ == "1.2.3.dev"
