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
from setuptools import find_packages, setup

# This reads the __version__ variable from cirq_google/_version.py
__version__ = ''
exec(open('cirq_google/_version.py').read())

name = 'cirq-google'

description = (
    'The Cirq module that provides tools and access to the Google Quantum Computing Service'
)

# README file as long_description.
long_description = open('README.rst', encoding='utf-8').read()

# If CIRQ_PRE_RELEASE_VERSION is set then we update the version to this value.
# It is assumed that it ends with one of `.devN`, `.aN`, `.bN`, `.rcN` and hence
# it will be a pre-release version on PyPi. See
# https://packaging.python.org/guides/distributing-packages-using-setuptools/#pre-release-versioning
# for more details.
if 'CIRQ_PRE_RELEASE_VERSION' in os.environ:
    __version__ = os.environ['CIRQ_PRE_RELEASE_VERSION']
    long_description = (
        "**This is a development version of Cirq-google and may be "
        "unstable.**\n\n**For the latest stable release of Cirq-google "
        "see**\n`here <https://pypi.org/project/cirq-google>`__.\n\n" + long_description
    )

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

# Sanity check
assert __version__, 'Version string cannot be empty'

# This is a pure metapackage that installs all our packages
requirements += [f'cirq-core=={__version__}']


packs = ['cirq_google'] + [
    'cirq_google.' + package for package in find_packages(where='cirq_google')
]

setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    python_requires=('>=3.10.0'),
    install_requires=requirements,
    license='Apache 2',
    description=description,
    long_description=long_description,
    packages=packs,
    package_data={
        'cirq_google': ['py.typed'],
        'cirq_google.api.v2': ['*'],
        'cirq_google.api.v1': ['*'],
        'cirq_google.devices.calibrations': ['*'],
        'cirq_google.devices.specifications': ['*'],
        'cirq_google.json_test_data': ['*'],
    },
)
