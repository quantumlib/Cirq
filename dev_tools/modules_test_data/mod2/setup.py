# pylint: disable=wrong-or-nonexistent-copyright-notice

import runpy

from setuptools import setup

name = 'module2'

__version__ = runpy.run_path('pack2/_version.py')['__version__']

setup(name=name, version=__version__, packages=['pack2'])
