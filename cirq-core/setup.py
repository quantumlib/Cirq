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

import runpy

from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = runpy.run_path('cirq/_version.py')['__version__']
assert __version__, 'Version string cannot be empty'

name = 'cirq-core'

description = (
    'A framework for creating, editing, and invoking '
    'Noisy Intermediate Scale Quantum (NISQ) circuits.'
)

# README file as long_description.
with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

# Read in requirements
with open('requirements.txt', encoding='utf-8') as file:
    requirements = [r.strip() for r in file]
with open('cirq/contrib/requirements.txt', encoding='utf-8') as file:
    contrib_requirements = [r.strip() for r in file]


cirq_packages = ['cirq'] + [
    'cirq.' + package for package in find_packages(where='cirq', exclude=['google', 'google.*'])
]

setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    maintainer="The Quantum AI open-source software maintainers",
    maintainer_email="quantum-oss-maintainers@google.com",
    python_requires='>=3.11.0',
    install_requires=requirements,
    extras_require={'contrib': contrib_requirements},
    license='Apache-2.0',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=cirq_packages,
    package_data={
        'cirq': ['py.typed'],
        'cirq.protocols.json_test_data': ['*'],
        f'cirq{"."}contrib.json_test_data': ['*'],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
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
