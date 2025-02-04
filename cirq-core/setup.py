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

import io
import os
from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ''
exec(open('cirq/_version.py').read())

name = 'cirq-core'

description = (
    'A framework for creating, editing, and invoking '
    'Noisy Intermediate Scale Quantum (NISQ) circuits.'
)

# README file as long_description.
long_description = io.open('README.md', encoding='utf-8').read()

# If CIRQ_PRE_RELEASE_VERSION is set then we update the version to this value.
# It is assumed that it ends with one of `.devN`, `.aN`, `.bN`, `.rcN` and hence
# it will be a pre-release version on PyPi. See
# https://packaging.python.org/guides/distributing-packages-using-setuptools/#pre-release-versioning
# for more details.
if 'CIRQ_PRE_RELEASE_VERSION' in os.environ:
    __version__ = os.environ['CIRQ_PRE_RELEASE_VERSION']
    long_description = (
        "<div align='center' width='50%'>\n\n"
        "| ⚠️ WARNING |\n"
        "|:----------:|\n"
        "| **This is a development version of `cirq-core` and may be<br>"
        "unstable. For the latest stable release of `cirq-core`,<br>"
        "please visit** <https://pypi.org/project/cirq-core>.|\n"
        "\n</div>\n\n" + long_description
    )

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]
contrib_requirements = open('cirq/contrib/requirements.txt').readlines()
contrib_requirements = [r.strip() for r in contrib_requirements]


cirq_packages = ['cirq'] + [
    'cirq.' + package for package in find_packages(where='cirq', exclude=['google', 'google.*'])
]

# Sanity check
assert __version__, 'Version string cannot be empty'

setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    maintainer="The Quantum AI open-source software maintainers",
    maintainer_email="quantum-oss-maintainers@google.com",
    python_requires=('>=3.10.0'),
    install_requires=requirements,
    extras_require={'contrib': contrib_requirements},
    license='Apache 2',
    description=description,
    long_description=long_description,
    packages=cirq_packages,
    package_data={'cirq': ['py.typed'], 'cirq.protocols.json_test_data': ['*']},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Quantum Computing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "algorithms",
        "api",
        "cirq",
        "google",
        "google quantum",
        "nisq",
        "python",
        "quantum",
        "quantum algorithms",
        "quantum circuit",
        "quantum circuit simulator",
        "quantum computer simulator",
        "quantum computing",
        "quantum development kit",
        "quantum information",
        "quantum programming",
        "quantum programming language",
        "quantum simulation",
        "sdk",
        "simulation",
    ],
)
