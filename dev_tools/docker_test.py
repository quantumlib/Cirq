import subprocess
import pathlib
import platform
import pytest


def test_docker():
    if platform.system() != 'Linux':
        pytest.skip("Unsupported os")
    root_folder = pathlib.Path(__file__).parent.parent
    buildResult = subprocess.run(['docker', 'build', '-t', 'cirq_image', '.'], cwd=root_folder)
    assert buildResult.returncode == 0

    result = subprocess.run(
        ["docker run cirq_image python -c \"import cirq; assert cirq.__version__ is not None\""],
        cwd=root_folder,
        shell=True,
    )
    assert result.returncode == 0
