from setuptools import setup

name = 'parent-module'

__version__ = ''

exec(open('mod1/pack1/_version.py').read())

setup(name=name, version=__version__, requirements=[])
