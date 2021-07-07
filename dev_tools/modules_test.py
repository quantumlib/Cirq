# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from dev_tools import modules
from dev_tools.modules import Module


def test_modules():
    mod1 = Module(
        root=Path('mod1'),
        raw_setup={
            'name': 'module1',
            'version': '0.12.0.dev',
            'url': 'http://github.com/quantumlib/cirq',
            'author': 'The Cirq Developers',
            'author_email': 'cirq-dev@googlegroups.com',
            'python_requires': '>=3.6.0',
            'install_requires': ['req1', 'req2'],
            'license': 'Apache 2',
            'packages': ['pack1', 'pack1.sub'],
        },
    )
    assert mod1.name == 'module1'
    assert mod1.version == '0.12.0.dev'
    assert mod1.top_level_packages == ['pack1']
    assert mod1.top_level_package_paths == [Path('mod1') / 'pack1']

    mod2 = Module(
        root=Path('mod2'), raw_setup={'name': 'module2', 'version': '1.2.3', 'packages': ['pack2']}
    )

    assert mod2.name == 'module2'
    assert mod2.version == '1.2.3'
    assert mod2.top_level_packages == ['pack2']
    assert mod2.top_level_package_paths == [Path('mod2') / 'pack2']
    assert modules.list_modules(search_dir=Path("dev_tools/modules_test_data")) == [mod1, mod2]

    parent = Module(
        root=Path('.'), raw_setup={'name': 'parent-module', 'version': '1.2.3', 'requirements': []}
    )
    assert parent.top_level_packages == []
    assert modules.list_modules(
        search_dir=Path("dev_tools/modules_test_data"), include_parent=True
    ) == [
        mod1,
        mod2,
        parent,
    ]


def test_cli():
    env = os.environ.copy()
    env["PYTHONPATH"] = "../.."
    output = subprocess.check_output(
        [sys.executable, "../modules.py", "list"],
        cwd="dev_tools/modules_test_data",
        env=env,
    )
    assert output.decode("utf-8") == "mod1 mod2 "


@contextlib.contextmanager
def chdir(*, target_dir: str = None):
    """Changes for the duration of the test the working directory.

    Args:
        target_dir: the target directory. If None is specified, it will create a temporary
            directory.
    """

    cwd = os.getcwd()
    tdir = tempfile.mkdtemp() if target_dir is None else target_dir
    os.chdir(tdir)
    try:
        yield
    finally:
        os.chdir(cwd)
        if target_dir is None:
            shutil.rmtree(tdir)


@chdir(target_dir="dev_tools/modules_test_data")
def test_main():
    with mock.patch('sys.stdout', new=StringIO()) as output:
        modules.main(["list", "--mode", "package-path"])
        assert output.getvalue() == ' '.join(
            [os.path.join("mod1", "pack1"), os.path.join("mod2", "pack2"), ""]
        )

    with mock.patch('sys.stdout', new=StringIO()) as output:
        modules.main(["list", "--mode", "folder", "--include-parent"])
        assert output.getvalue() == ' '.join(["mod1", "mod2", ".", ""])

    with mock.patch('sys.stdout', new=StringIO()) as output:
        modules.main(["list", "--mode", "package"])
        assert output.getvalue() == ' '.join(["pack1", "pack2", ""])


@chdir(target_dir=None)
def test_error():
    f = open("setup.py", mode='w')
    f.write('name="test"')
    f.close()

    with pytest.raises(AssertionError, match=r"Invalid setup.py - setup\(\) was not called.*"):
        modules.main(["list", "--mode", "folder", "--include-parent"])
