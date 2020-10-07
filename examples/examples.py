import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from astropy.table import Table
import astropy.units as u

from interferopy.cube import Cube, MultiCube
import interferopy.tools as iftools

# TODO: add outdir when all plots look good


def spectrum_single_pixel():
	"""
	Extract the spectrum from the central pixel.
	"""
	filename = "./data/Pisco.cube.50kms.image.fits"
	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy

	cub = Cube(filename)
	flux = cub.single_pixel_value()  # without coords gv=iven, use central pixel in the map
	# flux, _, _ = cub.spectrum()  # alternatively

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.plot(cub.freqs, flux * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(cub.freqs, flux * scale, 0, color="skyblue", step='mid', lw=0, alpha=0.3)
	ax.plot(cub.freqs, cub.rms * scale, color="gray", ls=":")  # noise levels
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Flux density (mJy / beam)")
	plt.savefig("./plots/spectrum_single_pixel.pdf", bbox_inches="tight")  # save plot
	plt.show()


def spectrum_aperture():
	"""
	Extract aperture spectrum at the specified position.
	"""
	filename = "./data/Pisco.cube.50kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # we know where the source is
	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy

	cub = Cube(filename)
	flux, err = cub.aperture_value(ra=ra, dec=dec, radius=1.3, calc_error=True)
	# flux, err _ = cub.spectrum(ra=ra, dec=dec, radius=1.3, calc_error=True)  # alternatively

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.plot(cub.freqs, flux * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(cub.freqs, flux * scale, 0, color="skyblue", step='mid', lw=0, alpha=0.3)
	ax.plot(cub.freqs, err * scale, color="gray", ls=":")  # 1sigma error
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Aperture flux density (mJy)")
	plt.savefig("./plots/spectrum_aperture.pdf", bbox_inches="tight")  # save plot
	plt.show()




def spectrum_aperture_paper():
	"""
	Compute residual corrected spectrum. Fit a Gaussian plus a continuum.
	Generate paper quality plot.
	"""
	filename = "./data/Pisco.cube.50kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # [degrees] we know where the source is
	radius = 1.3  # [arcsec] we know the size of the aperture we want
	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy

	# load the cube and perform residual scaling spectrum extraction
	mcub = MultiCube(filename)  # because the cubes follow a naming convention, will open several present cubes
	spectrum, err, tab = mcub.spectrum_corrected(ra=ra, dec=dec, radius=radius, calc_error=True)
	freqs = mcub.freqs  # this will be the x-axis

	# fit the spectrum with a Gaussian on top of a constant continuum, initial fit parameters (p0) must be set manually
	popt, pcov = curve_fit(iftools.gausscont, freqs, spectrum, p0=(1, 5, 222.5, 0.2), sigma=err, absolute_sigma=True)
	cont, amp, nu, sigma = popt
	cont_err, amp_err, nu_err, sigma_err = np.sqrt(np.diagonal(pcov))
	# compute some further numbers from the fit
	sigma_kms = iftools.ghz2kms(sigma, nu)
	fwhm_kms = iftools.sig2fwhm(sigma_kms)
	fwhm_err_kms = iftools.sig2fwhm(iftools.ghz2kms(sigma_err, nu))
	integral_fit = amp * sigma_kms * np.sqrt(2 * np.pi)
	integral_err = integral_fit * np.sqrt((sigma_err / sigma) ** 2 + (nu_err / nu) ** 2 + (amp_err / amp) ** 2)

	print("Gaussian fit:")
	print("Flux = " + str(iftools.sigfig(integral_fit, 2)) + " +- " + str(iftools.sigfig(integral_err, 1)) + " Jy.km/s")
	print("FWHM = " + str(iftools.sigfig(fwhm_kms, 2)) + " +- " + str(iftools.sigfig(fwhm_err_kms, 1)) + " km/s")
	print("Freq = " + str(iftools.sigfig(nu, 7)) + " +- " + str(iftools.sigfig(nu_err, 1)) + " GHz")

	# plot the spectrum, fill around the fitted continuum value
	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.plot(freqs, spectrum * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(freqs, spectrum * scale, cont * scale, color="skyblue", step='mid', lw=0, alpha=0.3)

	# plot Gaussian fit
	x_gauss = np.linspace(np.min(freqs), np.max(freqs), 1000)
	y_gauss = iftools.gausscont(x_gauss, *popt)
	ax.plot(x_gauss, y_gauss*scale, color="firebrick")

	# add velocity axis based around the fitted peak
	vels = mcub.cubes["image"].vels(nu)
	ax2 = ax.twiny()

	# match ranges of the two axes
	ax.set_xlim(freqs[0], freqs[-1])
	ax2.set_xlim(vels[0], vels[-1])

	# add axis labels
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Aperture flux density (mJy)")
	ax2.tick_params(direction='in', which="both")
	ax2.set_xlabel("Velocity (km s$^{-1}$)")

	# add the zero line
	ax.axhline(0, color="gray", lw=0.5, ls=":")

	plt.savefig("./plots/spectrum_aperture_paper.pdf", bbox_inches="tight")  # save plot
	plt.show()


def spectrum_aperture_technical():
	"""
	Compute residual corrected spectrum. Plot spectra extracted from various maps for comparison.
	Also plot clean-to-dirty beam ratio epsilon.
	"""
	filename = "./data/Pisco.cube.50kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # [degrees] we know where the source is
	radius = 1.3  # [arcsec] we know the size of the aperture we want
	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy

	# load the cube and perform residual scaling spectrum extraction
	mcub = MultiCube(filename)  # because the cubes follow a naming convention, will open several present cubes
	spectrum, err, tab = mcub.spectrum_corrected(ra=ra, dec=dec, radius=radius, calc_error=True)
	freqs = mcub.freqs  # this will be the x-axis

	# plot the spectrum, fill around the fitted continuum value
	fig, axes = plt.subplots( figsize=(4.8, 4.8),nrows=2, ncols=1,sharex=True, gridspec_kw={'height_ratios': [3, 1]})
	ax = axes[0]

	# the table returned from spectrum_corrected contains fluxes measured in different map
	# as well as clean-to-dirty beam ratios

	spectrum_dirty = tab["flux_dirty"]
	ax.plot(freqs, spectrum_dirty * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(freqs, spectrum_dirty * scale, 0, color="firebrick", step='mid', lw=0, alpha=1, label="Dirty")

	spectrum_uncorrected = tab["flux_image"]
	ax.plot(freqs, spectrum_uncorrected * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(freqs, spectrum_uncorrected * scale, 0, color="forestgreen", step='mid', lw=0, alpha=1,
					label="Uncorrected")

	ax.plot(freqs, spectrum * scale, color="black", drawstyle='steps-mid', lw=0.75)
	ax.fill_between(freqs, spectrum * scale, 0, color="skyblue", step='mid', lw=0, alpha=1, label="Corrected")


	ax.set_xlim(freqs[0], freqs[-1])
	ax.tick_params(direction='in', which="both")
	ax.set_ylabel("Aperture flux density (mJy)")
	ax.legend(frameon=False) # loc="upper right"

	ax2 = axes[1]
	# epsilon_fix was estimated from higher S/N channels and applied on all channels
	ax2.axhline(tab["epsilon_fix"][0], color="skyblue", lw=1)
	ax2.plot(freqs, tab["epsilon"], lw=0, marker="o", ms=1, color="black")

	ax2.tick_params(direction='in', which="both")
	ax2.set_xlabel("Frequency (GHz)")
	ax2.set_ylabel(r"$\epsilon$")
	ax2.set_ylim(-1.5,1.5)

	plt.savefig("./plots/spectrum_aperture_technical.pdf", bbox_inches="tight")  # save plot
	plt.show()


def main():
	spectrum_single_pixel()
	spectrum_aperture()
	spectrum_aperture_technical()
	spectrum_aperture_paper()


if __name__ == "__main__":
	main()
