'''
interferopy

Setup script
'''

# prefer setuptools over distutils, but fall back if not present
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='interferopy',
      version='0.1',
      author='Mladen Novak',
      author_email='novak@mpia.de',
      packages=['interferopy']
      )
