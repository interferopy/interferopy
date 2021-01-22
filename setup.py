'''
interferopy

Setup script
'''

from distutils.core import setup 
setup(name='interferopy',
	  version= '0.1',
	  author = 'Mladen Novak',
	  author_email = 'novak@mpia.de',
	  package_dir = {'interferopy':'src'},
      package_data = {'interferopy': ['examples/*']},
	  packages = ['interferopy']
	  )