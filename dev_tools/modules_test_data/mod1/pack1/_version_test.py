# pylint: disable=wrong-or-nonexistent-copyright-notice
import pack1._version  # type: ignore[import-not-found]


def test_version() -> None:
    assert pack1._version.__version__ == "1.2.3.dev"
