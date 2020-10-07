
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import astropy.units as u

from interferopy.cube import Cube, MultiCube
import interferopy.tools as iftools


def spectrum():
	cub = Cube("./data/Pisco.cube.50kms.image.fits")
	flux = cub.single_pixel_value()

	plt.plot(cub.freqs, flux)
	plt.show()


def main():
	spectrum()


if __name__ == "__main__":
	main()
