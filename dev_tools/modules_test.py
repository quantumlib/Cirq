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

import os
from io import StringIO
from unittest import mock

import pytest

from dev_tools import modules, shell_tools
from dev_tools.modules import Module


def test_modules():
    mod1 = Module(
        root='mod1',
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
    assert mod1.top_level_package_paths == [os.path.join('mod1', 'pack1')]

    mod2 = Module(
        root='mod2', raw_setup={'name': 'module2', 'version': '1.2.3', 'packages': ['pack2']}
    )

    assert mod2.name == 'module2'
    assert mod2.version == '1.2.3'
    assert mod2.top_level_packages == ['pack2']
    assert mod2.top_level_package_paths == [os.path.join('mod2', 'pack2')]
    assert modules.list_modules(search_dir="dev_tools/modules_test_data") == [mod1, mod2]

    parent = Module(
        root='.', raw_setup={'name': 'parent-module', 'version': '1.2.3', 'requirements': []}
    )
    assert parent.top_level_packages == []
    assert modules.list_modules(search_dir="dev_tools/modules_test_data", include_parent=True) == [
        mod1,
        mod2,
        parent,
    ]


def test_cli():
    output = shell_tools.output_of(
        "python3",
        "../modules.py",
        "list",
        cwd="dev_tools/modules_test_data",
        env={"PYTHONPATH": "../.."},
    )
    assert output == '\n'.join(["mod1", "mod2"])


def test_main():
    os.chdir("dev_tools/modules_test_data")
    with mock.patch('sys.stdout', new=StringIO()) as output:
        modules.main(["list", "--mode", "package-path"])
        assert output.getvalue() == '\n'.join(["mod1/pack1", "mod2/pack2", ""])

    with mock.patch('sys.stdout', new=StringIO()) as output:
        modules.main(["list", "--mode", "folder", "--include-parent"])
        assert output.getvalue() == '\n'.join(["mod1", "mod2", ".", ""])


def test_error(tmpdir_factory):
    cwd = tmpdir_factory.mktemp(basename="cirq_modules_test")
    os.chdir(cwd)

    f = open("setup.py", mode='w')
    f.write('name="test"')
    f.close()

    with pytest.raises(AssertionError, match="Invalid setup.py - setup\\(\\) was not called.*"):
        modules.main(["list", "--mode", "folder", "--include-parent"])
