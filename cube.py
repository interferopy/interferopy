import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
# import astropy.units as u
import interferopy.tools as tools
import scipy.constants as const


class Cube:
	# TODO: comment class attributes
	# TODO: try opening all maps: dirty, clean, model, psf
	# TODO: time calcrms on aspecs using .T and without .T

	def __init__(self, filename):
		"""
		Create a cube object by reading the fits image.
		:param filename: Path string to the fits image.
		"""
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
			# Compute above values, except rms, which is computed on demand.
			self.__load_fitsfile()
		else:
			raise FileNotFoundError(filename)

	def __load_fitsfile(self):
		"""
		Read the fits file and extract common useful data into Cube class attributes.
		"""
		self.log("Open " + self.filename)
		self.hdu = fits.open(self.filename)
		self.head = self.hdu[0].header
		self.pixsize = self.head["cdelt2"] * 3600  # arcsec, assumes square pixels

		# quick fix for a faulty header card that contained \n char - remove the tag altogether
		if "ORIGIN" in self.head.keys():
			self.head.remove('ORIGIN')

		# transpose the cube data so that intuitive im[ra,dec,freq,stokes] indexing can be used
		# number of cube dimensions
		naxis = len(self.hdu[0].data.shape)
		if naxis == 4 and self.hdu[0].data.shape[0] == 1:
			# drop the fourth (Stokes) axis if only a single plane is present
			image = self.hdu[0].data.T[:, :, :, 0]
			naxis = 3
		elif naxis == 4:
			image = self.hdu[0].data.T
			self.log("Warning: did test 4D full polarization cubes with multiple Stokes axes.")
		elif naxis == 3:
			image = self.hdu[0].data.T
		elif naxis == 2:
			# add the third axis, if missing, because several methods require it
			image = self.hdu[0].data[np.newaxis, :, :].T
			naxis = 3
			self.log("Warning: did not test 2D maps with the missing 3rd axis.")
		else:
			raise RuntimeError("Invalid number of cube dimensions.")
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
				else:
					# unknown
					reffreq = 0
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
			if len(self.im.shape) >= 3:
				nch = self.im.shape[2]
			else:
				nch = 1

			beam_table = Table()
			beam_table.add_columns([Table.Column(name='bmaj', data=np.ones(nch) * beam["bmaj"])])
			beam_table.add_columns([Table.Column(name='bmin', data=np.ones(nch) * beam["bmin"])])
			beam_table.add_columns([Table.Column(name='bpa', data=np.ones(nch) * beam["bpa"])])
			beam_table.add_columns([Table.Column(name='chan', data=range(nch))])
			beam_table.add_columns([Table.Column(name='pol', data=np.zeros(nch))])
		self.beam = beam_table

		# model image is Jy/pixel, and since beam volume divides integrated fluxes, set it to 1 in this case
		if "BUNIT" in self.head.keys() and self.head["BUNIT"].strip().lower().endswith("/pixel"):
			self.beamvol = np.array([1] * nch)
		else:
			# "BUNIT" in self.head.keys() and self.head["BUNIT"].strip().lower().endswith("/beam"):
			# sometimes BUNIT is not given, although beam is in the header (e.g. the PSF map)
			self.beamvol = np.pi / (4 * np.log(2)) * self.beam["bmaj"] * self.beam["bmin"] / self.pixsize ** 2
			if type(self.beamvol) is Table.Column:
				self.beamvol = self.beamvol.data

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
				self.__rms = np.array([tools.calcrms(self.im)])
		return self.__rms

	def set_rms(self, value):
		self.__rms = value

	rms = property(get_rms, set_rms)

	def deltavel(self, reffreq=None):
		"""
		Compute channel width in velocity units (km/s).
		:param reffreq: Computed around specific velocity. If empty, use referent one from the header.
		:return: Channel width in km/s. Sign reflects how channels are ordered.
		"""

		if reffreq is None:
			reffreq = self.reffreq

		return self.deltafreq / reffreq * const.c / 1000  # in km/s

	def vels(self, reffreq):
		"""
		Compute velocities of all cube channels for a given reference frequency.
		:param reffreq: Reference frequency in GHz. If empty, use referent one from the header.
		:return: Velocities in km/s.
		"""

		if reffreq is None:
			reffreq = self.reffreq

		return const.c / 1000 * (1 - self.freqs / reffreq)

	def radec2pix(self, ra=None, dec=None):
		"""
		Convert ra and dec coordinates into pixels to be used as im[px, py].
		If no coords are given, the center of the map is assumed.
		:param ra: Right Ascention in degrees.
		:param dec: Declination in degrees.
		:return: Coords x and y in pixels (0 based index).
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

		# need integer indices
		px = int(np.round(px))
		py = int(np.round(py))

		return px, py

	def pix2radec(self, px=None, py=None):
		"""
		Convert pixels coordinates into ra and dec.
		If no coords are given, the center of the map is assumed.
		:param px: X-axis index.
		:param py: Y-axis index.
		:return: Coords ra, dec in degrees
		"""
		if px is None or py is None:
			px = self.im.shape[0] / 2
			py = self.im.shape[1] / 2

		if len(self.wc.axis_type_names) < 3:
			ra, dec = self.wc.all_pix2world(px, py)
		else:
			ra, dec, _ = self.wc.all_pix2world(px, py, self.freqs[0], 0)

		return ra, dec

	def freq2pix(self, freq=None):
		"""
		Get the channel number of requested frequency.
		:param freq: Frequency in GHz.
		:return: Channel index.
		"""

		if len(self.wc.axis_type_names) < 3:
			raise ValueError("No frequency axis is present.")
			return None

		if freq is None:
			pz = self.im.shape[2] / 2
			pz = int(round(pz))
			return pz

		if freq < np.min(self.freqs) or freq > np.max(self.freqs):
			raise ValueError("Requested frequency is outside of the available range.")
			return None

		# This is ok, unless the 3rd axis is in velocities
		# ra0 = self.head["CRVAL1"]
		# dec0 = self.head["CRVAL2"]
		# _, _, pz = self.wc.all_world2pix(ra0, dec0, freq * 1e9, 0)
		# pz = int(np.round(pz))  # need integer index

		# freqs is populated when the cube is loaded, regardless whether the header is in vels or freqs
		# find the index with the minimum offset
		pz = np.argmin(np.abs(self.freqs - freq))

		return pz

	def pix2freq(self, pz=None):
		"""
		Get the frequency of a given channel.
		If no channel is given, the center channel is assumed.
		:param pz: Channel index.
		:return: Frequency in GHz.
		"""

		if len(self.wc.axis_type_names) < 3:
			raise ValueError("No frequency axis is present.")
			return None

		if pz is None:
			pz = self.im.shape[2] / 2
			pz = int(round(pz))

		if pz < 0 or pz >= len(self.freqs):
			raise ValueError("Requested channel is outside of the available range.")
			return None

		return self.freqs[pz]

	def distance_grid(self, px, py):
		"""
		Grid of distances from the chosen pixel. Uses small angle approximation (simple Pythagorean distances).
		Distances are measured in numbers of pixels.
		:param px: Index of x coord.
		:param py: Index of y coord.
		:return: 2D grid of distances in pixels, same shape as the cube slice
		"""

		xxx = np.arange(self.im.shape[0])
		yyy = np.arange(self.im.shape[1])
		distances = np.sqrt((yyy[np.newaxis, :] - py) ** 2 + (xxx[:, np.newaxis] - px) ** 2)
		return distances

	def spectrum(self, ra=None, dec=None, radius=0, channel=None, freq=None):
		"""
		Extract the spectrum (for 3D cube) or a single flux density value (for 2D map) at a given coord (ra, dec)
		integrated within a circular aperture of a given radius.
		If no coordinates are given, the center of the map is assumed.
		If no radius is given, a single pixel value is extracted (usual units Jy/beam), otherwise aperture
		integrated spectrum is extracted (units of Jy).
		Note: use the freqs field (or velocities method) to get the x-axis values.
		:param ra: Right Ascention in degrees.
		:param dec: Declination in degrees.
		:param radius: Circular aperture radius in arcsec.
		:param channel: Force extracton in a single channel of provided index (instead of the full cube).
		:param freq: Frequency in GHz, takes precedence over channel param.
		:return: spec, err, npix: spectrum, error estimate and number of pixels in the aperture
		"""

		px, py = self.radec2pix(ra, dec)
		if freq is not None:
			channel = self.freq2pix(freq)

		# take single pixel value if no aperture radius given
		if radius <= 0:
			self.log("Extracting single pixel spectrum.")
			spec = self.im[px, py, :]
			err = self.rms[:]  # single pixel error is just rms
			npix = np.ones(len(spec))
			# use just a single channel
			if channel is not None:
				spec = spec[channel]
				err = err[channel]
				npix = npix[channel]
		else:
			self.log("Extracting aperture spectrum.")
			# grid of distances from the source in arcsec, need for the aperture mask
			distances = self.distance_grid(px, py) * self.pixsize

			# select pixels within the aperture
			w = distances <= radius

			if channel is not None:
				npix = np.sum(np.isfinite(self.im[:, :, channel][w]))
				spec = np.nansum(self.im[:, :, channel][w]) / self.beamvol[channel]
				err = self.rms[channel] * np.sqrt(npix / self.beamvol[channel])
			else:
				spec = np.zeros(self.nch)
				npix = np.zeros(self.nch)
				for i in range(self.nch):
					spec[i] = np.nansum(self.im[:, :, i][w]) / self.beamvol[i]
					npix[i] = np.sum(np.isfinite(self.im[:, :, i][w]))
				err = self.rms * np.sqrt(npix / self.beamvol)

		if len(spec) == 1:
			spec = spec[0]
			err = err[0]
			npix = npix[0]

		return spec, err, npix

	def single_pixel_value(self, ra=None, dec=None, freq=None, channel=None):
		"""
		Get a single pixel value at the given coord.
		If freq is undefined, will return the spectrum of the 3D cube.
		Units: ra[deg], dec[deg], freq[GHz]
		Alias function. Check the "spectrum" method for details. The radius is fixed to 0 here.
		"""
		return self.spectrum(ra=ra, dec=dec, radius=0, freq=freq, channel=channel)

	def aperture_value(self, ra=None, dec=None, radius=1, freq=None, channel=None):
		"""
		Get an aperture integrated value at the given coord.
		If freq is undefined, will return the spectrum of the 3D cube.
		Units: ra[deg], dec[deg], radius[arcsec], freq[GHz]
		Alias function. Check the "spectrum" method for details. The radius is defaulted to 1 arcsec here.
		"""
		return self.spectrum(ra=ra, dec=dec, radius=radius, freq=freq, channel=channel)

	def growing_aperture(self, ra=None, dec=None, maxradius=1, binspacing=None, bins=None, channel=0, freq=None,
						 profile=False):
		"""
		Compute curve of growth at the given coordinate position in a circular aperture, growing up to the max radius.
		If no coordinates are given, the center of the map is assumed.
		If no frequency or channel is given, the whole spectrum is returned.
		:param ra: Right ascention in degrees.
		:param dec: Declination in degrees.
		:param maxradius: Max radius for aperture integration in arcsec.
		:param binspacing: Resolution of the growth flux curve in arcsec, default is one pixel size.
		:param bins: Custom bins for curve growth (1D np array).
		:param channel: Index of the cube channel to take.
		:param freq: Frequency in GHz, takes precedence over channel param.
		:param profile: If True, compute azimuthally averaged profile, if False, compute cumulative aperture values
		:return: radius, flux, err, npix - all 1D numpy arrays: aperture radius, cumulative flux within it,
		associated Poissionain error (based on number of beams inside the aprture and the map rms), number of pixels
		"""
		self.log("Running growth_curve.")

		# get coordinates in pixels
		px, py = self.radec2pix(ra, dec)
		# get grid of distances from the coordinate
		distances = self.distance_grid(px, py) * self.pixsize

		if freq is not None:
			channel = self.freq2pix(freq)

		if bins is None:
			if binspacing is None:
				# take one pixel size as default size
				binspacing = self.pixsize
			bins = np.arange(0, maxradius, binspacing)

		# histogram will fail if nans are present, select only valid pixels
		w = np.isfinite(self.im[:, :, channel])

		# histogram by default sums the number of pixels
		npix = np.histogram(distances[w], bins=bins)[0]
		if profile:
			pass
		else:
			npix = np.cumsum(npix)

		# pixel values are added as histogram weights to get the sum of pixel values
		flux = np.histogram(distances[w], bins=bins, weights=self.im[:, :, channel][w])[0]
		if profile:
			# mean value - azimuthally averaged
			flux = flux / npix
		else:
			# cumulative flux inside an aperture is the sum of all pixel values divided by the beam volume
			flux = np.cumsum(flux) / self.beamvol[channel]

		if profile:
			# error on the mean
			err = self.rms[channel] / np.sqrt(npix / self.beamvol)
		else:
			# error estimate assuming Poissonian statistics: rms x sqrt(number of independent beams inside aperture)
			err = self.rms[channel] * np.sqrt(npix / self.beamvol[channel])

		# centers of bins
		radius = 0.5 * (bins[1:] + bins[:-1])

		# old loop version, human readable, but super slow in execution for large apertures and lots of pixels
		# for i in range(len(bins)-1):
		# 	#w=(distances>=bins[i]) & (distances<bins[i+1]) #annulus
		# 	w=(distances>=0) & (distances<bins[i+1]) # aperture (cumulative)
		# 	npix[i]=np.sum(w)
		# 	flux[i]=np.sum(im[w])/beamvol
		# 	err[i]=rms*np.sqrt(npix[i]/beamvol) # rms times sqrt of beams used for integration

		return radius, flux, err, npix

	def aperture_r(self, ra=None, dec=None, maxradius=1.0, binspacing=None, bins=None, channel=0, freq=None):
		"""
		Obtain integrated flux within a circular aperture as a function of radius.
		If freq is undefined, will return the spectrum of the 3D cube.
		Alias function. Check the "growing_aperture" method for details.
		Units: ra[deg], dec[deg], maxradius[arcsec], binspacing[arcsec], freq[GHz]
		:return: radius, flux, err, npix
		"""
		return self.growing_aperture(ra=ra, dec=dec, maxradius=maxradius, binspacing=binspacing,
									 bins=bins, channel=channel, freq=freq, profile=False)

	def profile_r(self, ra=None, dec=None, maxradius=1.0, binspacing=None, bins=None, channel=0, freq=None):
		"""
		Obtain azimuthaly averaged profile as a function of radius.
		Alias function. Check the "growing_aperture" method for details.
		Units: ra[deg], dec[deg], maxradius[arcsec], binspacing[arcsec], freq[GHz]
		:return: radius, flux, err, npix
		"""
		return self.growing_aperture(ra=ra, dec=dec, maxradius=maxradius, binspacing=binspacing,
									 bins=bins, channel=channel, freq=freq, profile=True)

	def save_fitsfile(self, filename=None, overwrite=False):
		"""
		Save the cube in a fits file by storing the image and the header.
		:param filename: Path string to the output file. Uses input filename by default
		:param overwrite: False by default.
		:return:
		"""

		if filename is None and self.filename is None:
			raise ValueError("No filename was provided.")

		if not overwrite and filename is None:
			raise ValueError("Overwriting is disabled and no filename was provided.")

		# use the input filename if none is provided
		if overwrite and filename is None:
			filename = self.filename

		if os.path.exists(filename) and not overwrite:
			raise RuntimeError("Filename exist, but overwriting is disabled.")

		fits.writeto(filename, self.im.T, self.head, overwrite=overwrite)
		self.log("Fits file saved to " + filename)


class MultiCube:
	"""
	A container like class to hold multiple cubes at the same time. Cubes are stored in a dictionary.
	Example: mc = MultiCube("path_to_cube.image.fits") will load the cube, which is then accesible as mc["image"]

	"""

	def __init__(self, filename=None, autoload_multi=True):
		"""
		Provide the file path to the final cleaned cube. Will try to find other adjacent cubes based on their names.
		Standard key names from CASA are: image, residual, model, psf, pb, image.pbcor
		Additional names are dirty, clean.comp

		:param filename: Path string to the cleaned cube fits image.
		:param autoload_multi: If true, attempt to find other cubes using preset (mostly CASA) suffixes.
		"""

		# these are standard suffixes from CASA tclean output (except "dirty")
		# gildas output has different naming conventions, which are not implemnted here
		keylist = ["image", "residual", "dirty", "pb", "model", "psf", "image.pbcor", "clean.comp"]
		self.cubes = dict(zip(keylist, [None]*len(keylist)))
		self.basename = None

		if filename is None:
			# just setup the basic dictionary and exit
			return
		elif not os.path.exists(filename):
			raise FileNotFoundError(filename)

		filenames = dict(zip(keylist, [None]*len(keylist)))
		filenames["image"] = filename

		# TODO could improve searching, but there are multiple naming conventions
		# get the basename, which is hopefully shared between different cubes
		endings = [".image.tt0.fits", ".image.fits", ".fits"]
		for ending in endings:
			if filename.lower().endswith(ending):
				self.basename = filename[:-len(ending)]
				extension = ending.replace(".image", "")
				break

		# try to fill other filenames
		if autoload_multi:
			if self.basename is None:
				raise ValueError("Unknown extension.")

			# print("extension",extension)
			for k in keylist:
				# expected filename
				filename_suffixed = self.basename + "." + k + extension
				if os.path.exists(filename_suffixed):
					filenames[k] = filename_suffixed

		# load the cubes
		for k in self.cubes.keys():
			if filenames[k] is not None:
				self.cubes[k] = Cube(filenames[k])

		self.log("Loaded cubes: " + str(self.loaded_cubes))

	def load_cube(self, filename, key=None):
		"""
		Add provided fits file to the MultiCube instance.
		Known keys are: "image", "residual", "dirty", "pb", "model", "psf", "image.pbcor"
		If the filename does not end with these words, please provide a manual key.
		:param filename: Path string to the fits image.
		:param key: Dictionary key name used to store the cube.
		:return:
		"""
		# try to estimate the key from the filename
		if key is None:
			# assuming the file ends in something like .residual.fits
			suffix = str(filename).split(".")[-2]
			# if this suffix matches preset ones
			if suffix in self.cubes.keys():
				key = suffix

		# Load and store new cube
		cub = Cube(filename)

		if key is None:
			raise ValueError("Please provide key under which to store the cube.")

		if len(self.loaded_cubes) > 0:
			if cub.im.shape != self[self.loaded_cubes[0]].im.shape:
				self.log("Warning: Loaded cube has a different shape than existing one!")

		self[key] = cub

	def __getitem__(self, key):
		return self.cubes[key]

	def __setitem__(self, key, value):
		self.cubes[key] = value

	def get_loaded_cubes(self):
		"""
		Get a list of loaded cubes (those that are not None).
		:return: List of key names.
		"""
		return [k for k in self.cubes.keys() if self.cubes[k] is not None]

	loaded_cubes = property(get_loaded_cubes)

	def log(self, text):
		"""
		Basic logger function to allow better functionality in the future development.
		All class functions print info through this wrapper.
		Could be extended to provide different levels of info, timestamps, or logging to a file.
		"""
		print(text)

	def make_clean_comp(self, overwrite=False):
		"""
		Generate a clean component cube. Defined as the cleaned cube minus the residual, or, alternatively,
		model image convolved with the clean beam. This cube is not outputted by CASA.
		:param overwrite: If true, overrides any present "clean.comp" cube.
		:return: None
		"""
		self.log("Generate clean component cube.")

		if self["image"] is None:
			raise ValueError("Cleaned cube is missing.")
		if self["residual"] is None:
			raise ValueError("Residual cube is missing.")
		if self["clean.comp"] is not None and not overwrite:
			self.log("Warning: clean.comp cube already exists and overwriting is disabled.")

		cub = Cube(self["image"].filename)
		cub.im = self["image"].im - self["residual"].im
		cub.filename = None  # This cube is no longer the one on disk, so empty the filename as a precaution
		self["clean.comp"] = cub
		# return self["clean.comp"]

	def make_no_pbcorr(self, overwrite=False):
		"""
		Generate a flat noise cube from the primary beam (PB) corrected one, and the PB response (which is <= 1).
		PB corrected cube has valid fluxes, but rms computation is not straightforward.
		PB corrected cube = flat noise cube / PB response
		:param overwrite: If true, overrides any present "image"" cube.
		:return: None
		"""
		self.log("Generate flat noise image cube from the PB corrected one.")

		if self["image.pbcor"] is None:
			raise ValueError("PB corrected cleaned cube is missing.")
		if self["pb"] is None:
			raise ValueError("PB response cube is missing.")
		if self["image"] is not None and not overwrite:
			self.log("Warning: image cube already exists and overwriting is disabled.")

		cub = Cube(self["image.pbcor"].filename)
		cub.im = self["image.pbcor"].im * self["pb"].im
		cub.filename = None  # This cube is no longer the one on disk, so empty the filename as a precaution
		self["image"] = cub
		# return self["image"]

	def __cubes_prepare(self):
		"""
		Perform several prelimiaries before attempting residual scaled flux extraction.
		:return: True if no problems are detected, False otherwise.
		"""

		only_pbcor_exists = self["image"] is None and self["image.pbcor"] is not None and self["image.pb"] is not None
		if only_pbcor_exists:
			self.make_no_pbcorr()

		cubes_exists = self["image"] is not None and self["dirty"] is not None and self["residual"] is not None
		if not cubes_exists:
			self.log("Error: Need all three cubes: image, dirty, residual!")
			return False

		# generate clean component cube (this is not a standard CASA output)
		if self["clean.comp"] is None:
			self.make_clean_comp()

		shapes_equal = self["image"].im.shape == self["dirty"].im.shape == self["residual"].im.shape
		if not shapes_equal:
			self.log("Error: Cube shapes are not equal!")
			return False

		# Residual scaling assumes clean beam in all maps
		# Override the beam volume in dirty and residual maps to the cleaned one
		bvol = self["image"].beamvol
		self["dirty"].beamvol = bvol
		self["residual"].beamvol = bvol
		self["clean.comp"].beamvol = bvol

		return True

	def spectrum_corrected(self, ra=None, dec=None, radius=1.0, channel=None, freq=None):
		# residual scaling

		px, py = self["image"].radec2pix(ra, dec)
		if freq is not None:
			channel = self["image"].freq2pix(freq)

		# take single pixel value if no aperture radius given
		if radius <= 0:
			raise ValueError("No aperture is defined!")

		# check and prepare cubes for residual scaling
		if not self.__cubes_prepare():
			raise ValueError("Cubes check failed!")

		# could be optimized by sharing some arrays between runs, e.g. distance grid

		# run aperture extraction on all cubes
		params=dict(ra=ra, dec=dec, radius=radius, channel=channel, freq=freq)
		f = self["image"].spectrum(**params)
		c = self["clean.comp"].spectrum(**params)
		r = self["residual"].spectrum(**params)
		d = self["dirty"].spectrum(**params)

		# err=self["image"].rms*np.sqrt(npix/beamvol)
		#
		# epsilon=c/(d-r)
		# g=epsilon*d

		# TODO estimate epsilon from high S/N part
		# TODO add npix to spectrum?

		# return bins, f, c, r, d, epsilon, g, err, rms, npix

		# return spec

		return None

	def growing_aperture_corrected(self, ra=None, dec=None, maxradius=1, binspacing=None, bins=None,
								   channel=0, freq=None, profile=False):

		return None

