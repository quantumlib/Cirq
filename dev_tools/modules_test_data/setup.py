# pylint: disable=wrong-or-nonexistent-copyright-notice

import runpy

from setuptools import setup

name = 'parent-module'

__version__ = runpy.run_path('mod1/pack1/_version.py')['__version__']

setup(name=name, version=__version__, requirements=[])
