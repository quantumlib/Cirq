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

import io
import os
from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ''
exec(open('cirq_rigetti/_version.py').read())

name = 'cirq-rigetti'

description = 'A Cirq package to simulate and connect to Rigetti quantum computers and Quil QVM'

# README file as long_description.
long_description = io.open('README.rst', encoding='utf-8').read()

# If CIRQ_PRE_RELEASE_VERSION is set then we update the version to this value.
# It is assumed that it ends with one of `.devN`, `.aN`, `.bN`, `.rcN` and hence
# it will be a pre-release version on PyPi. See
# https://packaging.python.org/guides/distributing-packages-using-setuptools/#pre-release-versioning
# for more details.
if 'CIRQ_PRE_RELEASE_VERSION' in os.environ:
    __version__ = os.environ['CIRQ_PRE_RELEASE_VERSION']
    long_description = (
        "**This is a development version of Cirq-rigetti and may be "
        "unstable.**\n\n**For the latest stable release of Cirq-rigetti "
        "see**\n`here <https://pypi.org/project/cirq-rigetti>`__.\n\n" + long_description
    )

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

# Sanity check
assert __version__, 'Version string cannot be empty'

requirements += [f'cirq-core=={__version__}']

cirq_packages = ['cirq_rigetti'] + [
    'cirq_rigetti.' + package for package in find_packages(where='cirq_rigetti')
]


setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    python_requires='>=3.9.0',
    install_requires=requirements,
    license='Apache 2',
    description=description,
    long_description=long_description,
    packages=cirq_packages,
    package_data={'cirq_rigetti': ['py.typed'], 'cirq_rigetti.json_test_data': ['*']},
)
