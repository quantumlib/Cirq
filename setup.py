# Copyright 2018 The Cirq Developers
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
from setuptools import setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ''

from dev_tools import modules
from dev_tools.file_tools import read_file_filtered
from dev_tools.requirements import explode

exec(open('cirq-core/cirq/_version.py').read())

name = 'cirq'

description = (
    'A framework for creating, editing, and invoking '
    'Noisy Intermediate-Scale Quantum (NISQ) circuits.'
)

# Read README.rst for the long_description, skipping parts that contain rST
# constructs that are disallowed by PyPI.
long_description = read_file_filtered(
    "README.rst",
    ".. ▶︎─── start github-only",
    ".. ▶︎─── end github-only"
)

# If CIRQ_PRE_RELEASE_VERSION is set then we update the version to this value.
# The value is assumed to end with one of `.devN`, `.aN`, `.bN`, `.rcN`, which
# means this will be a pre-release version put on PyPI. See
# https://packaging.python.org/specifications/version-specifiers/#pre-releases
# for more details.
if 'CIRQ_PRE_RELEASE_VERSION' in os.environ:
    __version__ = os.environ['CIRQ_PRE_RELEASE_VERSION']
    long_description = (
        "**This is a development version of Cirq and may be "
        "unstable.**\n\n**For the latest stable release of Cirq "
        "see**\n`here <https://pypi.org/project/cirq>`__.\n\n" + long_description
    )

# Sanity check
assert __version__, 'Version string cannot be empty'

# This is a pure metapackage that installs all our packages
requirements = [f'{p.name}=={p.version}' for p in modules.list_modules()]

dev_requirements = explode('dev_tools/requirements/deps/dev-tools.txt')

# filter out direct urls (https://github.com/pypa/pip/issues/6301)
dev_requirements = [r.strip() for r in dev_requirements if "https://" not in r]

setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    python_requires='>=3.10.0',
    install_requires=requirements,
    extras_require={'dev_env': dev_requirements},
    license='Apache 2',
    description=description,
    long_description=long_description,
    packages=[],
)
