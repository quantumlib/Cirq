import subprocess
import pathlib
import pytest


@pytest.mark.docker
def test_docker():
    root_folder = pathlib.Path(__file__).parent.parent
    buildResult = subprocess.run(['docker', 'build', '-t', 'cirq_image', '.'], cwd=root_folder)
    assert buildResult.returncode == 0

    result = subprocess.run(
        ["docker run cirq_image python -c \"import cirq; assert cirq.__version__ is not None\""],
        cwd=root_folder,
        shell=True,
    )
    assert result.returncode == 0
