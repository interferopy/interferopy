import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import astropy.units as u


class Cube:
	# TODO: comment parameters

	def __init__(self, filename):
		if os.path.exists(filename):
			self.filename = filename
			self.hdu = None
			self.imhead = None
			self.beams = None
			self.wc = None
			self.image = None
			self.pixsize = None
			self.deltach = None

			self.__load_fitsfile()
		else:
			raise FileNotFoundError

	def log(self, text):
		print(text)

	def __load_fitsfile(self):
		self.log("Opening " + self.filename)

		hdu = fits.open(self.filename)
		imhead = hdu[0].header
		pixsize = imhead["cdelt2"] * 3600  # arcsec, assumes square pixels
		ndim = len(hdu[0].data.shape)  # number of cube dimensions

		# quick fix for a faulty header card that contained \n char - remove the tag altogether
		if "ORIGIN" in imhead.keys():
			imhead.remove('ORIGIN')

		# populate beam data
		beam = None
		beams = None
		# there is a beam per channel table present
		if len(hdu) > 1:
			beams = hdu[1].data
		# single beam in the main header
		elif "BMAJ" in imhead.keys() and "BMIN" in imhead.keys() and "BPA" in imhead.keys():
			beam = {"bmaj": imhead["BMAJ"] * 3600, "bmin": imhead["BMIN"] * 3600, "bpa": imhead["BPA"]}
			beams = beam

		# TODO: cleanup and ismplify these if-elses
		if ndim == 4 or ndim == 3:

			if ndim == 4 and hdu[0].data.shape[0] > 1:
				# TODO: implement special case when all Stokes planes are present
				# im=hdu[0].data.T
				raise Exception("Stokes 4D cubes not implemented.")
			if ndim == 4:
				im = hdu[0].data.T[:, :, :, 0]  # cube [ra,dec,freq] using stokes I
			if ndim == 3:
				im = hdu[0].data.T

			# not really a cube, single slice in freq
			if im.shape[2] == 1:
				im = im[:, :, 0]
				wc = wcs.WCS(imhead, naxis=2)
			# a 3D cube
			else:
				# create a table with beams for each channel when there is only one present
				if beam is not None:
					nch = im.shape[2]
					beams = Table()
					beams.add_columns([Table.Column(name='bmaj', data=np.ones(nch) * beam["bmaj"])])
					beams.add_columns([Table.Column(name='bmin', data=np.ones(nch) * beam["bmin"])])
					beams.add_columns([Table.Column(name='bpa', data=np.ones(nch) * beam["bpa"])])
					beams.add_columns([Table.Column(name='chan', data=range(nch))])
					beams.add_columns([Table.Column(name='pol', data=np.zeros(nch))])
				# grab existing beam table
				elif len(hdu) > 1:
					beams = hdu[1].data  # table of beams per ch
				# something unexpected
				else:
					beams = []

				# TODO: check noema datacubes, make sure deltas ar ok, add center as well
				if str(imhead["CTYPE3"]).strip().lower() == "vrad":
					deltach = imhead["cdelt3"] * 1e-3  # km/s
				elif str(imhead["CTYPE3"]).strip().lower() == "freq":
					deltach = imhead["cdelt3"] * 1e-9  # GHz
				else:
					deltach = 0

				wc = wcs.WCS(imhead, naxis=3)

		elif ndim == 2:
			im = hdu[0].data.T
			wc = wcs.WCS(imhead, naxis=2)

		else:
			raise Exception("The cube does not have between 2 and 4 dimensions.")

		# if not beams:
		# 	self.log("No beam sizes found.")

		self.hdu = hdu
		self.imhead = imhead
		self.beams = beams
		self.wc = wc
		self.image = im
		self.pixsize = pixsize
		self.deltac = deltach


# def get_filename(self):
# 	return self._filename
# filename = property(get_filename)


class MultiCube:
	def __init__(self):
		raise NotImplementedError
