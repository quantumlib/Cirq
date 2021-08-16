from setuptools import setup

name = 'module2'

__version__ = ''


exec(open('pack2/_version.py').read())

setup(name=name, version=__version__, packages=['pack2'])
