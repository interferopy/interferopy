'''
interferopy

Setup script
'''
# prefer setuptools over distutils, but fall back if not present
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt', "r") as f:
    requirements = f.readlines()


setup(name='interferopy',
      version='0.5',
      author='Leindert Boogaard, Romain Meyer, Mladen Novak',
      author_email='novak@mpia.de',
      description='python library for common tasks in radio/submm interferometric data analysis',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/mladenovak/interferopy",
      project_urls={
          "Bug Tracker": "https://github.com/mladenovak/interferopy/issues"
      },
      python_requires=">=3.7",
      install_requires=requirements,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      packages=['interferopy']
      )
