import os
from copy import deepcopy
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord


def sigfig(x, digits=2):
	"""
	Round a number to a number of significant digits.
	:param x: Input number.
	:param digits: Number of significant digits.
	:return:
	"""
	if x == 0:
		return 0
	else:
		return round(x, digits - int(np.floor(np.log10(abs(x)))) - 1)


def weighted_avg(values, weights):
	"""
	Compute weighted average of a masked array. Used in computing mean amplitude vs uv from flagged data.
	:param values: Input masked array.
	:param weights: Input weights.
	:return: average, standard_error, standard_deviation
	"""
	average = np.ma.average(values, weights=weights)
	variance = np.ma.average((values - average) ** 2, weights=weights)
	standard_deviation = np.ma.sqrt(variance)
	standard_error = standard_deviation / np.ma.sqrt(sum(weights))  # need -1?
	return average, standard_error, standard_deviation


def gauss(x, a, mu, sigma):
	"""
	Gaussian profile. Not normalized.
	:param x: Values of x where to compute the profile.
	:param a: Gaussian amplitude (peak height).
	:param mu: Gaussian center (peak position).
	:param sigma: Gaussian sigma (width).
	:return: Value at x.
	"""
	return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gausscont(x, b, a, mu, sigma):
	"""
	Gaussian profile (not normalized) on top of a constant continuum.
	:param x: Values of x where to compute the profile.
	:param b: Constant offset (in y-axis direction).
	:param a: Gaussian amplitude (peak height).
	:param mu: Gaussian center (peak position).
	:param sigma: Gaussian sigma (width).
	:return: Value at x.
	"""
	return b + gauss(x, a, mu, sigma)


def fwhm2sig(fwhm):
	"""
	Convert Gaussian FWHM to sigma.
	"""
	return np.abs(fwhm) / (2 * np.sqrt(2 * np.log(2)))


def sig2fwhm(sigma):
	"""
	Convert Gaussian sigma to FWHM.
	"""
	return np.abs(sigma) * (2 * np.sqrt(2 * np.log(2)))


def kms2mhz(width_kms, freq_ghz):
	"""
	Convert channel width in km/s into MHz at the specified reference frequency.
	:param width_kms:
	:param freq_ghz:
	:return:
	"""
	return width_kms / (const.c / 1000) * freq_ghz * 1000  # widthmhz


def mhz2kms(width_mhz, freq_ghz):
	"""
	Convert channel width in MHz into km/s at the specified reference frequency.
	:param width_mhz:
	:param freq_ghz:
	:return:
	"""
	return width_mhz / (freq_ghz * 1000) * (const.c / 1000)  # widthkms


def kms2ghz(width_kms, freq_ghz):
	"""
	Convert channel width in km/s into GHz at the specified reference frequency.
	:param width_kms:
	:param freq_ghz:
	:return:
	"""
	return width_kms / (const.c / 1000) * freq_ghz  # widthghz


def ghz2kms(width_ghz, freq_ghz):
	"""
	Convert channel width in GHz into km/s at the specified reference frequency.
	:param width_ghz:
	:param freq_ghz:
	:return:
	"""
	return width_ghz / freq_ghz * (const.c / 1000)  # widthkms


def calcrms(arr, fitgauss=False, around_zero=True, clip_sigma=3, maxiter=20):
	"""
	Calculate rms by iteratively disregarding outlier pixels (beyond clip_sigma x rms values).
	:param arr: Input array.
	:param fitgauss: If True, Gaussian will be fitted onto the distribution of negative pixels.
	:param around_zero: Assume no systematic offsets, i.e., noise oscillates around zero.
	:param clip_sigma: Clip values beyond these many sigmas (in both positive and negative directions).
	:param maxiter: Maximum number of iterations to perform.
	:return: rms or, if fitgauss is Truem rms and Gaussian sigma
	"""
	a = arr[np.isfinite(arr)].flatten()
	rms = 0

	# TODO: likely a similar function already exists in some statistical library (look for Chauvenet)

	# iteratively calc rms and remove all values outside 3sigma
	mu = 0
	n = 0
	for i in range(maxiter):
		# mean of 0 is expected for noise
		if not around_zero:
			mu = np.nanmean(a)

		if len(a) > 0:
			rms = np.sqrt(np.mean(a ** 2))
		else:
			# rms = 0
			break

		w = (a > mu - clip_sigma * rms) & (a < mu + clip_sigma * rms)
		if n == np.sum(w):
			break
		a = a[w]
		n = np.sum(w)

	# alternative noise estimation: fit a Gaussian to the negative part of the pixel distribution
	# uses previous rms result as the intiial fit guess (p0)
	if fitgauss:
		bins = np.linspace(-10 * rms, 0, 100)
		bincen = 0.5 * (bins[1:] + bins[:-1])
		h = np.histogram(arr, bins=bins)[0]

		# mirror negative part for fitting
		xxx = np.append(bincen, -1 * bincen[::-1])
		yyy = np.append(h, h[::-1])

		popt, pcov = curve_fit(lambda x, a, sigma: gauss(x, a, 0, sigma), xxx, yyy, p0=(np.max(yyy), rms))
		a, sigma = popt
		sigma = np.abs(sigma)

		return rms, sigma
	else:
		return rms


def beam_volume_sr(bmaj, bmin=None):
	"""
	Compute Gaussian beam volume.
	:param bmaj: Major axis FWHM in arcsec.
	:param bmin: Minor axis FWHM in arcsec. If not provided, will assume circular beam bmin=bmaj.
	:return: Beam volume in steradians.
	"""

	# bmaj and bmin are major and minor FWHMs of a Gaussian (synthesised) beam, in arcsec
	if bmin is None:
		bmin = bmaj

	omega = np.pi / 4 / np.log(2) * (bmaj / 3600 / 180 * np.pi) * (bmin / 3600 / 180 * np.pi)  # convert to radians
	return omega


def surf_temp(freq, rms, theta):
	"""
	Copmute surface brightness temperature sensitivity in Kelvins. Used to compare radio map sensitivities.
	:param freq: Observed frequency in GHz.
	:param rms: Noise in the map in Jy/beam.
	:param theta: Beam FWHM in arcsec. Assumes circular beam.
	:return: Temperature in Kelvins.
	"""

	temp = rms * 1e-26 / beam_volume_sr(theta) * const.c ** 2 / (2 * const.k * (freq * 1e9) ** 2)
	return temp


def blackbody(nu, temp):
	"""
	Planck's law for black body emission, per unit frequency.
	:param nu: Rest frame frequency in Hz.
	:param temp: Temperature in K.
	:return: Emission in units of W / (m^2 Hz)
	"""
	return 2 * const.h * nu ** 3 / const.c ** 2 / (np.exp(const.h * nu / (const.k * temp)) - 1)


def dust_lum(nu_rest, Mdust, Tdust, beta):
	"""
	Compute intrinsic dust luminosity at specific rest frame frequency assuming modified black body emission.
	:param nu_rest: Rest frame frequency in Hz.
	:param Mdust: Total dust mass in kg.
	:param Tdust:  Dust temperature in K.
	:param beta: Emissivity coefficient, dimensionless.
	:return: Luminosity (at rest frequency nu) in W/Hz
	"""

	# dust opacity from Dunne+2003
	kappa_ref = 2.64  # m**2/kg
	kappa_nu_ref = const.c / 125e-6  # Hz

	# Dunne+2000 ?
	# kappa_ref=0.77*u.cm**2/u.g
	# kappa_ref=kappa_ref.to(u.m**2/u.kg).value
	# kappa_nu_ref=c/850e-6

	lum_nu = 4 * const.pi * kappa_ref * (nu_rest / kappa_nu_ref) ** beta * Mdust * blackbody(nu_rest, Tdust)
	return lum_nu


def dust_sobs(nu_obs, z, mass_dust, temp_dust, beta, cmb_contrast=True, cmb_heating=True):
	"""
	Compute observed flux density of the dust continuum, assuming a modified black body.
	:param nu_obs: Observed frame frequency in Hz.
	:param z: Redshift of the source.
	:param mass_dust: Total dust mass in kg.
	:param temp_dust: Intrinsic dust temperature in K (the source would have at z=0).
	:param beta: Emissivity coefficient, dimensionless.
	:param cmb_contrast: Correct for cosmic microwave background contrast.
	:param cmb_heating: Correct for cosmic microwave background heating (important at high z where Tcmb ~ Tdust).
	:return: Observed flux density in W/Hz/m^2.
	"""

	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
	dl = cosmo.luminosity_distance(z).to(u.m).value  # m
	nu_rest = nu_obs * (1 + z)
	temp_cmb0 = 2.73  # cmb temperature at z=0
	temp_cmb = (1 + z) * temp_cmb0

	# cmb heating (high redshift) and contrast corrections from da Cunha+2013
	if cmb_heating:
		temp_dustz = (temp_dust ** (4 + beta) + temp_cmb0 ** (4 + beta) * ((1 + z) ** (4 + beta) - 1)) ** (1 / (4 + beta))
	else:
		temp_dustz = temp_dust

	if cmb_contrast:
		f_cmb = 1. - blackbody(nu_rest, temp_cmb) / blackbody(nu_rest, temp_dustz)
	else:
		f_cmb = 1

	flux_obs = f_cmb * (1 + z) / (4 * np.pi * dl ** 2) * dust_lum(nu_rest, mass_dust, temp_dustz, beta)

	return flux_obs


def stack2d(ras, decs, im, imhead, imrms=None, pathout=None, overwrite=False, naxis=100, interpol=True):
	"""
	Perform median and mean (optionally rms weighted) stacking of multiple sources in a single radio map.
	This function requires that the first index is the x coordinate, and the second one y.
	If the map was opened with the fits package, it likely has to be transposed first with im.T.
	Opening the map as interferopy.Cube object does this transposition automatically.

	:param ras: List of right ascentions.
	:param decs: List of declinations.
	:param im: 2D radio map indexed as im[ra,dec].
	:param imhead: Image header.
	:param imrms: 2D rms noise map; if provided the mean stack will be noise weigthed.
	:param pathout: Filename for the output stacks, if None, no output is written to disk.
	:param overwrite: Overwrites files when saving (pathout).
	:param naxis: The size of the cutout square in pixels.
	:param interpol: Perform subpixel inteprolation (bicubic spline)
	:return: stack_mean, stack_median, stack_head, cube - 2D map, 2D map, text header, 3D cube of cutouts
	"""

	# how many objects
	n = len(np.atleast_1d(ras))

	# calculate half of the cutout size
	halfxis = naxis / 2
	# one pixel too much in even sized, only odd naxis can have a "central" pixel
	even_shift = -1 if naxis % 2 == 0 else 0

	# get coord system
	wc = wcs.WCS(imhead, naxis=2)
	# pixel coords
	pxs, pys = wc.all_world2pix(ras, decs, 0)

	# allocate space for the stack cube
	cube = np.full((naxis, naxis, n), np.nan)
	cuberms = np.full((naxis, naxis, n), 1.0)

	# fill the cube with (interpolated) cutouts
	for i in range(n):
		# source position in pixels (float)
		px = np.atleast_1d(pxs)[i]
		py = np.atleast_1d(pys)[i]

		if interpol:
			# offset between the source and pixel centres
			xoff = px - int(px)
			yoff = py - int(py)
			# truncate source center for array positioning for interpolation
			px = int(px)
			py = int(py)
		else:
			xoff = 0
			yoff = 0
			# round the pixel to closest integer when not interpolating
			px = int(round(px))
			py = int(round(py))

		# calculate different offsets for cropping (edge effects)
		left = int(px - halfxis)
		right = int(px + halfxis + even_shift)
		bottom = int(py - halfxis)
		top = int(py + halfxis + even_shift)
		crop_left = int(-min([left, 0]))
		crop_right = int(min([im.shape[0] - 1 - right, 0]))
		crop_bottom = int(-min([bottom, 0]))
		crop_top = int(min([im.shape[1] - 1 - top, 0]))

		# initialize cutout
		subim = np.full((naxis, naxis), np.nan)
		subimrms = np.full((naxis, naxis), 1.0)

		# check if the cutout is at least partially inside the map
		not_in_map = abs(crop_left) >= naxis or abs(crop_right) >= naxis \
					 or abs(crop_bottom) >= naxis or abs(crop_top) >= naxis

		if not not_in_map:
			# get cutout from the map, nans are left where no data is available
			# interpolate so that stacking coord is in the middle pixel
			# cut if cutout is at least partially inside the map

			subim[0 + crop_left:naxis - 1 + crop_right + 1, 0 + crop_bottom:naxis - 1 + crop_top + 1] = \
				im[left + crop_left:right + crop_right + 1, bottom + crop_bottom:top + crop_top + 1]
			if imrms is not None:
				subimrms[0 + crop_left:naxis - 1 + crop_right + 1, 0 + crop_bottom:naxis - 1 + crop_top + 1] = \
					imrms[left + crop_left:right + crop_right + 1, bottom + crop_bottom:top + crop_top + 1]

			if interpol:
				# nans are not handled in interpolation, replace them temporarily with 0
				w = np.isnan(subim)
				subim[w] = 0
				subimrms[w] = 1.0

				# bicubic spline
				f = interp2d(range(naxis), range(naxis), subim, kind='cubic')
				subim = f(np.array(range(naxis)) + yoff, np.array(range(naxis)) + xoff)
				subim[w] = np.nan

				if imrms is not None:
					f2 = interp2d(range(naxis), range(naxis), subimrms, kind='cubic')
					subimrms = f2(np.array(range(naxis)) + yoff, np.array(range(naxis)) + xoff)
					subimrms[w] = np.nan

		# other possible interpolation methods:

		# griddata
		# x = np.linspace(0,naxis-1,naxis)
		# xcoo, ycoo = np.meshgrid(x,x)
		# xcof, ycof = np.meshgrid(x+yoff,x+xoff)
		# subim=griddata( (xcoo.flatten(), ycoo.flatten()), subim.flatten(), (xcof, ycof), method='cubic')
		# subimrms=griddata( (xcoo.flatten(), ycoo.flatten()), subimrms.flatten(), (xcof, ycof), method='cubic')

		# Rectangular B spline
		# narr=range(naxis)
		# sp = interpolate.RectBivariateSpline(narr, narr, subim, kx=4, ky=4, s=0)
		# subim=sp(narr+xoff, narr+yoff)
		# sp = interpolate.RectBivariateSpline(narr, narr, subimrms, kx=4, ky=4, s=0)
		# subimrms=sp(narr+xoff, narr+yoff)

		# add the sourcecutout to the cube
		cube[:, :, i] = subim
		cuberms[:, :, i] = subimrms

	# colapse the cube to mean and median stacks while handling NaNs properly
	cube_masked = np.ma.MaskedArray(cube, mask=~(np.isfinite(cube)))
	stack_mean = np.ma.average(cube_masked, weights=cuberms ** (-2), axis=2).filled(np.nan)
	stack_median = np.ma.median(cube_masked, axis=2).filled(np.nan)

	# edit the header to proper axis sizes
	stack_head = deepcopy(imhead)
	stack_head['NAXIS1'] = naxis
	stack_head['NAXIS2'] = naxis
	stack_head['CRPIX1'] = naxis / 2
	stack_head['CRPIX2'] = naxis / 2

	# save files
	if pathout is not None:
		# remove the extension to append a suffix for different files
		basepath = os.path.splitext(pathout)[0]
		# transpose back for proper writing (numpy index ordering rule)
		fits.writeto(basepath + '_median.fits', stack_median.T, stack_head, overwrite=overwrite)
		fits.writeto(basepath + '_mean.fits', stack_mean.T, stack_head, overwrite=overwrite)
		fits.writeto(basepath + '_cube.fits', cube.T, stack_head, overwrite=overwrite)
	# fits.writeto(basepath+'_cuberms.fits',cuberms.T, stack_head, clobber=True)

	return stack_mean, stack_median, stack_head, cube


def sex2deg(ra_hms, dec_dms, frame='icrs'):
	"""
	Convert sexagesimal coords (hours:minutes:seconds, degrees:minutes:seconds) to degrees.
	:param ra_hms: Right ascentions. String or list of strings.
	:param dec_dms: Declinations. String or list of strings.
	:param frame: Equinox frame. ALMA default is ICRS.
	:return: ra_deg, dec_deg - 1D numpy arrays with coords in degrees
	"""

	# wrap single value in a list for iteration
	single_val = False
	if isinstance(ra_hms, str) and isinstance(dec_dms, str):
		ra_hms = [ra_hms]
		dec_dms = [dec_dms]
		single_val = True

	n = len(ra_hms)
	ra_deg = np.zeros(n)
	dec_deg = np.zeros(n)

	for i in range(n):
		coo = SkyCoord(str(ra_hms[i]) + " " + str(dec_dms[i]), frame=frame, unit=(u.hourangle, u.deg))
		ra_deg[i] = coo.ra.deg
		dec_deg[i] = coo.dec.deg

	if single_val:
		return ra_deg[0], dec_deg[0]
	else:
		return ra_deg, dec_deg


def arcsec2kpc(z: float = 0):
	"""
	Return the number of kiloparsecs contained in one arcsecond at given redshift.
	Use concordance cosmology.
	:param z: Redshift of the source
	:return:
	"""
	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
	return 1. / cosmo.arcsec_per_kpc_proper(z).value