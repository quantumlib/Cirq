from setuptools import find_packages, setup

# This reads the __version__ variable from cirq/_version.py
__version__ = ''
exec(open('pack1/_version.py').read())

name = 'module1'


# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

pack1_packages = ['pack1'] + ['pack1.' + package for package in find_packages(where='pack1')]

# Sanity check
assert __version__, 'Version string cannot be empty'

setup(
    name=name,
    version=__version__,
    url='http://github.com/quantumlib/cirq',
    author='The Cirq Developers',
    author_email='cirq-dev@googlegroups.com',
    python_requires=('>=3.9.0'),
    install_requires=requirements,
    license='Apache 2',
    packages=pack1_packages,
)
