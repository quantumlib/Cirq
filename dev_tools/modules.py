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
import dataclasses
import os
import sys
from typing import List, Dict, Any

_FOLDER = 'folder'
_PACKAGE_PATH = 'package-path'


@dataclasses.dataclass
class Module:
    root: str
    raw_setup: Dict[str, Any]

    name: str = dataclasses.field(init=False)
    version: str = dataclasses.field(init=False)
    top_level_packages: List[str] = dataclasses.field(init=False)

    def __post_init__(self):
        self.name = self.raw_setup['name']
        if 'packages' in self.raw_setup:
            self.top_level_packages = [p for p in self.raw_setup['packages'] if '.' not in p]
        else:
            self.top_level_packages = []
        self.top_level_package_paths = [os.path.join(self.root, p) for p in self.top_level_packages]
        self.version = self.raw_setup['version']


def list_modules(
    search_dir: str = os.path.join(os.path.dirname(__file__), '..'), include_parent: bool = False
) -> List[Module]:
    """Returns a list of python modules based defined by setup.py files.

    Args:
        include_parent: if true, a setup.py is expected in `search_dir`, and the corresponding
        module will be included.
        search_dir: the search directory for modules, by default the repo root.
    Returns:
          a list of `Module`s that were found, where each module `m` is initialized with `m.root`
           relative to `search_dir`, `m.raw_setup` contains the dictionary equivalent to the
           keyword args passed to the `setuptools.setup` method in setup.py
    """

    relative_folders = sorted(
        [
            f
            for f in os.listdir(search_dir)
            if os.path.isdir(os.path.join(search_dir, f))
            and os.path.isfile(os.path.join(search_dir, f, "setup.py"))
        ]
    )
    if include_parent:
        parent_setup_py = os.path.join(search_dir, "setup.py")
        assert os.path.isfile(
            parent_setup_py
        ), f"include_parent=True, but {parent_setup_py} does not exist."
        relative_folders.append(".")

    result = [
        Module(root=folder, raw_setup=_parse_module(os.path.join(search_dir, folder)))
        for folder in relative_folders
    ]

    return result


def _parse_module(folder: str) -> Dict[str, Any]:
    setup_args = {}
    import setuptools

    orig_setup = setuptools.setup
    cwd = os.getcwd()

    def setup(**kwargs):
        nonlocal setup_args
        setup_args = kwargs

    try:
        setuptools.setup = setup
        os.chdir(folder)
        setup_py = open("setup.py").read()
        exec(setup_py, globals())
        assert setup_args, f"Invalid setup.py - setup() was not called in {folder}/setup.py!"
        return setup_args
    except BaseException as ex:
        print(f"Failed to run {folder}/setup.py:")
        raise ex
    finally:
        setuptools.setup = orig_setup
        os.chdir(cwd)


def _print_list_modules(mode: str, include_parent: bool = False):
    """Prints certain properties of cirq modules on separate lines.

    Module root folder and top level package paths are supported. The search dir is the current
    directory.

    Args:
        mode: 'folder' lists the root folder for each module, 'package-path' lists the path to
         the top level package(s).
        include_cirq: when true the cirq metapackage is included in the list
    Returns:
          a list of strings
    """
    for m in list_modules(".", include_parent):
        if mode == _FOLDER:
            print(m.root, end=" ")
        elif mode == _PACKAGE_PATH:
            for p in m.top_level_package_paths:
                print(p, end=" ")


def main(argv: List[str]):
    args = parse(argv)
    f = args.func
    delattr(args, 'func')
    f(**vars(args))


def parse(args):
    parser = argparse.ArgumentParser('A utility for modules.')
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
        "--include-parent",
        help="whether to include the parent package or not",
        default=False,
        action="store_true",
    )
    list_modules_cmd.set_defaults(func=_print_list_modules)
    return parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv[1:])  # coverage: ignore
