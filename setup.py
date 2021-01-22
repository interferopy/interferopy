'''
Created on July 2, 2018

@author: Romain A. Meyer

Setup script
'''

from distutils.core import setup 
setup(name='QUICFit',
	  version= '0.2',
	  author = 'Romain A. Meyer',
	  author_email = 'r.meyer.17@ucl.ac.uk',
	  package_dir = {'QUICFit':'src'},
      package_data = {'QUICFit': ['data/*.txt']},
	  packages = ['QUICFit']
	  )