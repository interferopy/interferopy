import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


# TODO: add more detailed comments and param descriptions

# round a number to a given number of significant digits
def sigfig(x, digits=2):
	if x == 0:
		return 0
	else:
		return round(x, digits - int(np.floor(np.log10(abs(x)))) - 1)


# get weighted average from masked arrays
def weighted_avg(values, weights):
	average = np.ma.average(values, weights=weights)
	variance = np.ma.average((values-average)**2, weights=weights)
	standard_deviation = np.ma.sqrt(variance)
	standard_error = standard_deviation / np.ma.sqrt(sum(weights))  # need -1?
	return average, standard_error, standard_deviation


# Gaussian profile
def gauss(x, a, mu, sigma):
	# a=gauss amplitude
	# mu=center
	# sigma=gauss sigma
	return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Gaussian profile plus a constant vertical offset (e.g., line on top of simple continuum)
def gausscont(x, b, a, mu, sigma):
	# b=const offset
	# a=gauss amplitude
	# mu=center
	# sigma=gauss sigma
	return b + gauss(x, a, mu, sigma)


# Gaussian widths conversions
def fwhm2sig(fwhm):
	return fwhm / (2 * np.sqrt(2 * np.log(2)))
def sig2fwhm(sigma):
	return sigma * (2 * np.sqrt(2 * np.log(2)))


# Channel width coversions between velocity and frequency (using radio velocity definition)
def kms2mhz(width_kms, freq_ghz):
	return width_kms/(const.c/1000) * freq_ghz * 1000  # widthmhz
def mhz2kms(width_mhz, freq_ghz):
	return width_mhz / (freq_ghz * 1000) * (const.c/1000)  # widthkms
def kms2ghz(width_kms, freq_ghz):
	return width_kms/(const.c/1000) * freq_ghz  # widthghz
def ghz2kms(width_ghz, freq_ghz):
	return width_ghz / freq_ghz * (const.c/1000)  # widthkms


# Calculate rms in the map (disregarding outlier pixels)
def calcrms(arr, fitgauss=False, around_zero=True, clip_sigma=3, maxiter=20):
	a = arr[np.isfinite(arr)].flatten()
	rms = 0

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

	if fitgauss:
		return rms, sigma
	else:
		return rms


# get Gaussian beam volume in sr
def beam_volume_sr(bmaj, bmin=None):
	# bmaj and bmin are major and minor FWHMs of a Gaussian (synthesised) beam, in arcsec
	if bmin is None:
		bmin = bmaj

	omega = np.pi/4/np.log(2)*(bmaj/3600/180*np.pi) * (bmin/3600/180*np.pi)  # convert to radians

	return omega


# get surface brightness temperature sensitivity in Kelvins
def surf_temp(freq, sigma, theta):
	# freq in GHz
	# sigma = rms noise in Jy/beam
	# theta = resolution FWHM in arcsec

	temp = sigma * 1e-26 / beam_volume_sr(theta) * const.c ** 2 / (2 * const.k * (freq * 1e9) ** 2)

	return temp


# black body radiation (Planck's law)
def blackbody(nu, T):
	return 2*const.h*nu**3/const.c**2 / (np.exp(const.h*nu/(const.k*T))-1)


# get intrinsic dust luminosity at specific rest frame frequency
def dust_lum(nu_rest, Mdust, Tdust, beta):
	# nu in Hz - rest frame frequency
	# Mdust in kg - total dust mass
	# Tdust in K - dust temperature
	# beta no dim
	# returns luminosity (at rest frequency nu) in W/Hz

	# dust opacity from Dunne+2003
	kappa_ref = 2.64  # m**2/kg
	kappa_nu_ref = const.c/125e-6  # Hz

	# Dunne+2000 ?
	# kappa_ref=0.77*u.cm**2/u.g
	# kappa_ref=kappa_ref.to(u.m**2/u.kg).value
	# kappa_nu_ref=c/850e-6

	lum_nu = 4*const.pi*kappa_ref*(nu_rest/kappa_nu_ref)**beta * Mdust * blackbody(nu_rest, Tdust)

	return lum_nu


# get observed flux density of the dust continuum, assuming a modified black body
def dust_sobs(nu_obs, z, Mdust, Tdust, beta, cmb_contrast=True, cmb_heating=True):
	# nu_obs in Hz - observed frame frequency
	# Mdust in kg - total dust mass
	# beta - modified black body parameter
	# Tdust is intrinisic dust temperature (at z=0)
	# returns observed flux density in W/Hz/m^2

	cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
	DL = cosmo.luminosity_distance(z).to(u.m).value # m
	nu_rest = nu_obs*(1+z)
	Tcmb0 = 2.73 # cmb temperature at z=0
	Tcmb = (1+z)*Tcmb0

	# cmb heating (high redshift) and contrast corrections from da Cunha+2013
	if cmb_heating:
		Tdustz = (Tdust**(4+beta)+Tcmb0**(4+beta)*((1+z)**(4+beta)-1))**(1/(4+beta))
	else:
		Tdustz = Tdust

	if cmb_contrast:
		f_cmb = 1.-blackbody(nu_rest, Tcmb)/blackbody(nu_rest, Tdustz)
	else:
		f_cmb = 1

	S_obs = f_cmb * (1+z) / (4*np.pi*DL**2) * dust_lum(nu_rest, Mdust, Tdustz, beta)

	return S_obs