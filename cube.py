import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import astropy.units as u
import interferopy.tools as tools
import scipy.constants as const


class Cube:
	# TODO: comment class attributes
	# TODO: try opening all maps: dirty, clean, model, psf
	# TODO: time calcrms on aspecs using .T and without .T

	def __init__(self, filename):
		if os.path.exists(filename):
			self.filename = filename
			self.hdu = None
			self.head = None
			self.im = None
			self.wc = None
			self.beam = None
			self.beamvol = None
			self.pixsize = None
			self.nch = None
			self.freqs = None
			self.reffreq = None
			self.deltafreq = None
			self.__rms = None
			self.__load_fitsfile()  # Computes most of above values
		else:
			raise FileNotFoundError

	def __load_fitsfile(self):
		"""
		Read the fits file and extract common useful data into Cube class atributtes.
		"""
		self.log("Opening " + self.filename)
		self.hdu = fits.open(self.filename)
		self.head = self.hdu[0].header
		self.pixsize = self.head["cdelt2"] * 3600  # arcsec, assumes square pixels

		# quick fix for a faulty header card that contained \n char - remove the tag altogether
		if "ORIGIN" in self.head.keys():
			self.head.remove('ORIGIN')

		# transpose the cube data so that intuitive im[ra,dec,freq] indexing can be used
		# drop the fourth (Stokes) axis if only a single plane is present
		naxis = len(self.hdu[0].data.shape)  # number of cube dimensions
		if naxis == 4 and self.hdu[0].data.shape[0] == 1:
			image = self.hdu[0].data.T[:, :, :, 0]
			naxis = 3
		elif 2 <= naxis <= 4:
			image = self.hdu[0].data.T
			if naxis == 4 and self.hdu[0].data.shape[0] > 1:
				self.log("Warning: did not experiment with full polarization cubes.")
		else:
			raise Exception("Invalid number of cube dimensions.")
		self.im = image
		self.naxis = naxis

		# save the world coord system
		self.wc = wcs.WCS(self.head, naxis=naxis)

		# populate frequency details (freq array, channel size, number of channels)
		# convert from velocity header if necessary, scale to GHz
		nch = 1
		if naxis >= 3:
			nch = self.im.shape[2]  # number of (freq) channels in the cube
			# if frequencies are already 3rd axis
			if str(self.head["CTYPE3"]).strip().lower() == "freq":
				_, _, freqs = self.wc.all_pix2world(int(self.im.shape[0] / 2), int(self.im.shape[1] / 2), range(nch), 0)
				freqs *= 1e-9  # in GHz
				self.deltafreq = self.head["CDELT3"] * 1e-9
				self.reffreq = self.head["CRVAL3"] * 1e-9
			# if frequencies are given in radio velocity, convert to freqs
			elif str(self.head["CTYPE3"]).strip().lower() == "vrad":
				_, _, vels = self.wc.all_pix2world(int(self.im.shape[0] / 2), int(self.im.shape[1] / 2), range(nch), 0)
				if "RESTFREQ" in self.head.keys():
					reffreq = self.head["RESTFREQ"] * 1e-9
				elif "RESTFRQ" in self.head.keys():
					reffreq = self.head["RESTFRQ"] * 1e-9
				freqs = reffreq * (1 - vels / const.c)  # in GHz
				self.reffreq = reffreq

				if "CUNIT3" in self.head.keys() and self.head["CUNIT3"].strip().lower() == "km/s":
					vel_scale = 1000
				else:
					vel_scale = 1
				self.deltafreq = reffreq * (self.head["CDELT3"] * vel_scale / const.c)
			else:
				freqs = None
				self.log("Warning: unknown 3rd axis format.")
			self.freqs = freqs
		self.nch = nch

		# populate beam data
		# single beam from the main header
		if "BMAJ" in self.head.keys() and "BMIN" in self.head.keys() and "BPA" in self.head.keys():
			beam = {"bmaj": self.head["BMAJ"] * 3600, "bmin": self.head["BMIN"] * 3600, "bpa": self.head["BPA"]}
		else:
			beam = {"bmaj": 0, "bmin": 0, "bpa": 0}

		# if there is a beam per channel table present, take it
		if nch > 1 and len(self.hdu) > 1:
			beam_table = self.hdu[1].data  # this is inserted by CASA, there is unit metadata, but should be arcsec
		# otherwise clone the single beam across all channels in a new table
		else:
			nch = self.im.shape[2]
			beam_table = Table()
			beam_table.add_columns([Table.Column(name='bmaj', data=np.ones(nch) * beam["bmaj"])])
			beam_table.add_columns([Table.Column(name='bmin', data=np.ones(nch) * beam["bmin"])])
			beam_table.add_columns([Table.Column(name='bpa', data=np.ones(nch) * beam["bpa"])])
			beam_table.add_columns([Table.Column(name='chan', data=range(nch))])
			beam_table.add_columns([Table.Column(name='pol', data=np.zeros(nch))])
		self.beam = beam_table

		# model image is Jy/pixel, and since beam volume divides integrated fluxes, set it to 1 in this case
		if "BUNIT" in self.head.keys() and self.head["BUNIT"].strip().lower().endswith("/pixel"):
			self.beamvol = 1
		else:
			self.beamvol = np.pi / (4 * np.log(2)) * self.beam["bmaj"] * self.beam["bmin"] / self.pixsize ** 2
			if type(self.beamvol) is Table.Column:
				self.beamvol = self.beamvol.data
			if nch == 1:
				self.beamvol = self.beamvol[0]


	def log(self, text):
		"""
		Basic logger function to allow better functionality in the future development.
		All class functions print info through this wrapper.
		Could be extended to provide different levels of info, timestamps, or logging to a file.
		"""
		print(text)

	def get_rms(self):
		"""
		Calculate rms for each channel of the cube. Can take some time on large cubes.
		:return: single rms value if 2D, array odf rms values for each channel if 3D cube
		"""
		if self.__rms is None:
			# calc rms per channel
			if self.nch > 1:
				self.__rms = np.zeros(self.nch)
				self.log("Computing rms of each channel.")
				for i in range(self.nch):
					self.__rms[i] = tools.calcrms(self.im[:, :, i])
			else:
				self.log("Computing rms.")
				self.__rms = tools.calcrms(self.im)
		return self.__rms

	def set_rms(self, value):
		self.__rms = value

	rms = property(get_rms, set_rms)

	def deltavel(self, reffreq=None):
		"""
		Computes channel width in velocity units (km/s).
		:param reffreq: Computed around specific velocity. If empty, will use referent one from the header.
		:return: Channel width in km/s. Sign reflects how channels are ordered.
		"""
		if reffreq is None:
			reffreq = self.reffreq

		return self.deltafreq / reffreq * const.c / 1000  # in km/s

	def vels(self, reffreq):
		"""
		Compute velocities of all cube channels for a given reference frequency.
		:param reffreq: Reference frequency in GHz.
		:return: Velocities in km/s.
		"""
		return const.c / 1000 * (1 - self.freqs / reffreq)

	def spectrum(self, ra=None, dec=None, radius=0, channel=None):
		"""
		Extract the spectrum (for 3D cube) or a single flux density value (for 2D map) at a given coord (ra, dec)
		integrated within a circular aperture of a given radius.
		If no coordinates are given, the central pixel is assumed.
		If no radius is given, a single pixel value is extracted (usual units Jy/beam), otherwise aperture
		integrated spectrum is extracted (units of Jy).
		Note: use the freqs field (or velocities method) to get the x-axis values.
		:param ra: Right Ascention in degrees.
		:param dec: Declination in degrees.
		:param radius: Circular aperture radius in arcsec.
		:param channel: Force extracton in a single channel of provided index (instead of the full cube).
		:return: Spectrum as 1D array. Alternatively,  a single value if 2D map was loded, or single channel chosen.
		"""

		# use the central pixel if no coords given
		if ra is None or dec is None:
			px = self.im.shape[0] / 2
			py = self.im.shape[1] / 2
		# otherwise convert radec to pixel coord
		else:
			if len(self.wc.axis_type_names) < 3:
				px, py = self.wc.all_world2pix(ra, dec)
			else:
				px, py, _ = self.wc.all_world2pix(ra, dec, self.freqs[0], 0)

		px = int(np.round(px))
		py = int(np.round(py))

		# take single pixel value if no aperture radius given
		if radius <= 0:
			spec = self.im[px, py, :]
			# use just a single channel
			if channel is not None:
				spec = spec[channel]
		else:
			# grid of distances from the source, need for the aperture mask
			xxx = np.arange(self.im.shape[0])
			yyy = np.arange(self.im.shape[1])
			distances = np.sqrt((yyy[np.newaxis, :] - py) ** 2 + (xxx[:, np.newaxis] - px) ** 2) * self.pixsize

			# select pixels within the aperture
			w = distances <= radius

			if channel is not None:
				spec = np.nansum(self.im[:, :, channel][w]) / self.beamvol[channel]
			else:
				spec = np.zeros(self.nch)
				for i in range(self.nch):
					spec[i] = np.nansum(self.im[:, :, i][w]) / self.beamvol[i]

		if len(spec == 1):
			spec = spec[0]

		return spec

	def growth_curve(self, ra=None, dec=None, maxradius=0, channel=0):
		# single channel curve of growth, up to max radius
		# add profile capability or put into separate function

		# TODO

		return None


class MultiCube:
	def __init__(self):
		raise NotImplementedError

	# need image, residual, dirty for residual scaling - beamvol must be overriden in this case
	# add model and psf specific things?
