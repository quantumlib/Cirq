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

import pathlib
import runpy
import sys

from setuptools import setup

# Ensure we can import our own dev_tools
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from dev_tools import modules
from dev_tools.requirements import explode

# This reads the __version__ variable from cirq-core/cirq/_version.py
__version__ = runpy.run_path('cirq-core/cirq/_version.py')['__version__']
assert __version__, 'Version string cannot be empty'

name = 'cirq'

description = (
    'A framework for creating, editing, and invoking '
    'Noisy Intermediate Scale Quantum (NISQ) circuits.'
)

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

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
    maintainer="Google Quantum AI open-source maintainers",
    maintainer_email="quantum-oss-maintainers@google.com",
    python_requires='>=3.11.0',
    install_requires=requirements,
    extras_require={'dev_env': dev_requirements},
    license='Apache-2.0',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=[],
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
