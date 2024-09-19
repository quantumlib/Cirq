# pylint: disable=wrong-or-nonexistent-copyright-notice
import pack1._version


def test_version():
    assert pack1._version.__version__ == "1.2.3.dev"
