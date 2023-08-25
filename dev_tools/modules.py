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


"""Utility tool for cirq modules.

It can be used as a python library for python scripts as well as a CLI tool for
bash scripts and interactive use.

Features:

listing modules:
 - Python: see list_modules

Version management:
 - Python: get_version and replace_version
 - CLI:
    - python3 dev_tools/modules.py print_version
    - python3 dev_tools/modules.py replace_version --old v0.12.0.dev --new v0.12.1.dev

optional arguments:
  -h, --help            show this help message and exit

subcommands:
  valid subcommands

  {list,print_version,replace_version}
                        additional help
    list                lists all the modules
    print_version       Check that all module versions are the same, and print it.
    replace_version     replace Cirq version in all modules
"""

import argparse
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

_FOLDER = 'folder'
_PACKAGE_PATH = 'package-path'
_PACKAGE = 'package'

_DEFAULT_SEARCH_DIR: Path = Path(".")


@dataclasses.dataclass
class Module:
    root: Path
    raw_setup: Dict[str, Any]

    name: str = dataclasses.field(init=False)
    version: str = dataclasses.field(init=False)
    top_level_packages: List[str] = dataclasses.field(init=False)
    top_level_package_paths: List[Path] = dataclasses.field(init=False)
    install_requires: List[str] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.name = self.raw_setup['name']
        if 'packages' in self.raw_setup:
            self.top_level_packages = [p for p in self.raw_setup['packages'] if '.' not in p]
        else:
            self.top_level_packages = []
        self.top_level_package_paths = [self.root / p for p in self.top_level_packages]
        self.version = self.raw_setup['version']
        self.install_requires = (
            [] if 'install_requires' not in self.raw_setup else self.raw_setup['install_requires']
        )


def list_modules(
    search_dir: Path = _DEFAULT_SEARCH_DIR, include_parent: bool = False
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
    Raises:
        ValueError: if include_parent=True but there is no setup.py in `search_dir`.
    """

    relative_folders = sorted(
        f.relative_to(search_dir)
        for f in search_dir.glob("*")
        if f.is_dir() and (f / "setup.py").is_file()
    )
    if include_parent:
        parent_setup_py = search_dir / "setup.py"
        if not parent_setup_py.exists():
            raise ValueError(f"include_parent=True, but {parent_setup_py} does not exist.")
        relative_folders.append(Path("."))

    result = [
        Module(root=folder, raw_setup=_parse_module(search_dir / folder))
        for folder in relative_folders
    ]

    return result


def get_version(search_dir: Path = _DEFAULT_SEARCH_DIR) -> Optional[str]:
    """Check for all versions are the same and return that version.

    Lists all the modules within `search_dir` (default the current working directory), checks that
    all of them are the same version and returns that version. If no modules found, None is
    returned, if more than one, ValueError is raised.

    Args:
        search_dir: the search directory for modules.
    Returns:
        None if no modules are found, the version number if exactly one version number is found.
    Raises:
        ValueError: if more than one version numbers are found.
    """
    try:
        mods = list_modules(search_dir=search_dir, include_parent=True)
    except ValueError:
        return None
    versions = {m.name: m.version for m in mods}
    if len(set(versions.values())) > 1:
        raise ValueError(f"Versions should be the same, instead: \n{versions}")
    return list(set(versions.values()))[0]


def replace_version(search_dir: Path = _DEFAULT_SEARCH_DIR, *, old: str, new: str):
    """Replaces the current version number with a new version number.

    Args:
        search_dir: the search directory for modules.
        old: the current version number.
        new: the new version number.
    Raises:
        ValueError: if `old` does not match the current version, or if there is not exactly one
            version number in the found modules.
    """
    version = get_version(search_dir=search_dir)
    if version != old:
        raise ValueError(f"{old} does not match current version: {version}")

    _validate_version(new)

    for m in list_modules(search_dir=search_dir, include_parent=True):
        version_file = _find_version_file(search_dir / m.root)
        content = version_file.read_text("UTF-8")
        new_content = content.replace(old, new)
        version_file.write_text(new_content)


def _validate_version(new_version: str):
    if not re.match(r"\d+\.\d+\.\d+(\.dev)?", new_version):
        raise ValueError(f"{new_version} is not a valid version number.")


def _find_version_file(top: Path) -> Path:
    for root, _, files in os.walk(str(top)):
        if "_version.py" in files:
            return Path(root) / "_version.py"
    raise FileNotFoundError(f"Can't find _version.py in {top}.")


def _parse_module(folder: Path) -> Dict[str, Any]:
    setup_args = {}
    import setuptools

    orig_setup = setuptools.setup
    cwd = os.getcwd()
    sys.path.insert(0, cwd)

    def setup(**kwargs):
        setup_args.update(kwargs)

    try:
        setuptools.setup = setup
        os.chdir(str(folder))
        setup_py = open("setup.py").read()
        exec(setup_py, globals(), {})
        assert setup_args, f"Invalid setup.py - setup() was not called in {folder}/setup.py!"
        return setup_args
    except BaseException:
        print(f"Failed to run {folder}/setup.py:")
        raise
    finally:
        setuptools.setup = orig_setup
        sys.path.remove(cwd)
        os.chdir(cwd)


############################################
# CLI MANAGEMENT
############################################


# --------------
# print_version
# --------------


def _print_version():
    print(get_version())


def _add_print_version_cmd(subparsers):
    print_version_cmd = subparsers.add_parser(
        "print_version", help="Check that all module versions are the same, and print it."
    )
    print_version_cmd.set_defaults(func=_print_version)


# --------------
# replace_version
# --------------


def _replace_version(old: str, new: str):
    replace_version(old=old, new=new)
    print(f"Successfully replaced version {old} with {new}.")


def _add_replace_version_cmd(subparsers):
    replace_version_cmd = subparsers.add_parser(
        "replace_version", help="replace Cirq version in all modules"
    )
    replace_version_cmd.add_argument(
        "--old", required=True, help="the current version to be replaced"
    )
    replace_version_cmd.add_argument("--new", required=True, help="the new version to be replaced")
    replace_version_cmd.set_defaults(func=_replace_version)


# --------------
# list_modules
# --------------


def _add_list_modules_cmd(subparsers):
    list_modules_cmd = subparsers.add_parser("list", help="lists all the modules")
    list_modules_cmd.add_argument(
        "--mode",
        default=_FOLDER,
        choices=[_FOLDER, _PACKAGE_PATH, _PACKAGE],
        type=str,
        help="'folder' to list root folder for module (e.g. cirq-google),\n"
        "'package-path' for top level python package path (e.g. cirq-google/cirq_google),\n"
        "'package' for top level python package (e.g cirq_google),\n",
    )
    list_modules_cmd.add_argument(
        "--include-parent",
        help="whether to include the parent package or not",
        default=False,
        action="store_true",
    )
    list_modules_cmd.set_defaults(func=_print_list_modules)


def _print_list_modules(mode: str, include_parent: bool = False):
    """Prints certain properties of cirq modules on separate lines.

    Module root folder and top level package paths are supported. The search dir is the current
    directory.

    Args:
        mode: 'folder' lists the root folder for each module, 'package-path' lists the path to
         the top level package(s).
        include_parent: when true the cirq metapackage is included in the list
    """
    for m in list_modules(Path("."), include_parent):
        if mode == _FOLDER:
            print(m.root, end=" ")
        elif mode == _PACKAGE_PATH:
            for p in m.top_level_package_paths:
                print(p, end=" ")
        elif mode == _PACKAGE:
            for package in m.top_level_packages:
                print(package, end=" ")


def parse(args):
    parser = argparse.ArgumentParser('A utility for modules.')
    subparsers = parser.add_subparsers(
        title='subcommands', description='valid subcommands', help='additional help'
    )
    _add_list_modules_cmd(subparsers)
    _add_print_version_cmd(subparsers)
    _add_replace_version_cmd(subparsers)
    return parser.parse_args(args)


def main(argv: List[str]):
    if argv == []:
        # If no arguments are given, print the help/usage info.
        argv = ['--help']
    args = parse(argv)
    # args.func is where we store the function to be called for a given subparser
    # e.g. it is list_modules for the `list` subcommand
    f = args.func
    # however the func is not going to be needed for the function itself, so
    # we remove it here
    del args.func
    f(**vars(args))


if __name__ == '__main__':
    main(sys.argv[1:])  # pragma: no cover
