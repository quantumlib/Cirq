import pytest
import subprocess
from python_ci_utils import env_tools


def test_failure_and_success():
    assert env_tools.sub_run(
        'true',
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'true',
        raise_error_if_process_fails=True,
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'false',
        raise_error_if_process_fails=False,
        silent=True
    ) == (None, None, 1)

    with pytest.raises(subprocess.CalledProcessError):
        env_tools.sub_run(
            'false',
            raise_error_if_process_fails=True,
            silent=True)


def test_capture_stdout():
    assert env_tools.sub_run(
        'echo', 'test',
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test',
        capture_stdout=False,
        silent=True
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test',
        capture_stdout=True,
        silent=True
    ) == ('test\n', None, 0)


def test_capture_shell_stderr():
    assert env_tools.sub_run(
        'echo', 'test', '1>&2',
        silent=True,
        shell=True,
    ) == (None, None, 0)

    assert env_tools.sub_run(
        'echo', 'test', '1>&2',
        silent=True,
        shell=True,
        capture_stderr=True,
    ) == (None, 'test\n', 0)
