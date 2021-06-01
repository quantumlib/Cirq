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
import argparse
import os
import sys
from typing import Union

_FOLDER = 'folder'
_PACKAGE_PATH = 'package-path'


def list_modules(mode: Union[_FOLDER, _PACKAGE_PATH], include_cirq: bool):
    """Prints the cirq modules on separate lines.

    Args:
        mode: 'folder' lists the root folder for each module, 'package-path' lists the path to the
            top level package(s).
        include_cirq: when true the cirq metapackage is included in the list
    Returns:
          none, instead prints results on the stdout
    """
    folders = sorted(
        [
            f
            for f in os.listdir('.')
            if os.path.isdir(f) and os.path.isfile(os.path.join(f, "setup.py"))
        ]
    )
    if include_cirq:
        folders.append('.')
    if mode == _FOLDER:
        for folder in folders:
            print(folder)
    elif mode == _PACKAGE_PATH:
        for folder in folders:
            packages = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f))
                and os.path.isfile(os.path.join(folder, f, "__init__.py"))
            ]
            for package in packages:
                print(package)


def main(args):
    repo_root = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(repo_root)
    f = args.func
    delattr(args, 'func')
    f(**vars(args))


def parse(args):
    parser = argparse.ArgumentParser('Monorepo utility.')
    subparsers = parser.add_subparsers(
        title='subcommands', description='valid subcommands', help='additional help'
    )
    list_modules_cmd = subparsers.add_parser("list", help="lists all the modules")
    list_modules_cmd.add_argument(
        "--mode",
        default=_FOLDER,
        choices=[_FOLDER, _PACKAGE_PATH],
        type=str,
        help="'folder' to list root folder for module,\n"
        "'package-path' for top level python package path",
    )
    list_modules_cmd.add_argument(
        "--include-cirq",
        help="whether to include the cirq metapackage or not",
        default=False,
        type=bool,
    )
    list_modules_cmd.set_defaults(func=list_modules)
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse(sys.argv[1:]))
