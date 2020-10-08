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
	ax.set_title("Single pixel spectrum")
	ax.plot(cub.freqs, flux * scale, color="black", drawstyle='steps-mid', lw=0.75, label="Spectrum")
	ax.fill_between(cub.freqs, flux * scale, 0, color="skyblue", step='mid', lw=0, alpha=0.3)
	ax.plot(cub.freqs, cub.rms * scale, color="gray", ls=":", label="Noise rms")
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Flux density (mJy / beam)")
	ax.legend(frameon=False)
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
	aper=1.3
	flux, err = cub.aperture_value(ra=ra, dec=dec, radius=aper, calc_error=True)
	# flux, err _ = cub.spectrum(ra=ra, dec=dec, radius=1.3, calc_error=True)  # alternatively

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.set_title("Integrated aperture spectrum")
	ax.plot(cub.freqs, flux * scale, color="black", drawstyle='steps-mid', lw=0.75, label="Spectrum within r="+str(aper)+'"')
	ax.fill_between(cub.freqs, flux * scale, 0, color="skyblue", step='mid', lw=0, alpha=0.3)
	ax.plot(cub.freqs, err * scale, color="gray", ls=":", label=r"1$\sigma$ error")  # 1sigma error
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Frequency (GHz)")
	ax.set_ylabel("Aperture flux density (mJy)")
	ax.legend(frameon=False)
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

	# Plot the uncorrected specturum as well
	# ax.plot(freqs, tab["flux_image"] * scale, color="black", drawstyle='steps-mid', lw=0.5, ls="--")

	# plot Gaussian fit
	x_gauss = np.linspace(freqs[0], freqs[-1], 1000)
	y_gauss = iftools.gausscont(x_gauss, *popt)
	ax.plot(x_gauss, y_gauss * scale, color="firebrick")

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
	ax2.set_xlabel(r"Velocity (km s$^{-1}$)")

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

	# tab.write("spectrum.txt", format="ascii.fixed_width", overwrite=True)  # save results in a human readable format

	# plot the spectrum, fill around the fitted continuum value
	fig, axes = plt.subplots(figsize=(4.8, 4.8), nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
	ax = axes[0]
	ax.set_title("Spectrum with and without correction")

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
	ax.legend(frameon=False)  # loc="upper right"

	ax2 = axes[1]
	# epsilon_fix was estimated from higher S/N channels and applied on all channels
	ax2.axhline(tab["epsilon_fix"][0], color="skyblue", lw=1)
	ax2.plot(freqs, tab["epsilon"], lw=0, marker="o", ms=1, color="black")

	ax2.tick_params(direction='in', which="both")
	ax2.set_xlabel("Frequency (GHz)")
	ax2.set_ylabel(r"Clean-to-dirty\nbeam ratio: $\epsilon$")
	ax2.set_ylim(-1.5, 1.5)

	plt.savefig("./plots/spectrum_aperture_technical.pdf", bbox_inches="tight")  # save plot
	plt.show()


def growing_aperture():
	"""
	Compute azimuthally averaged profile and the curve of growth (cumulative flux in aperture) as a function of radius.
	"""

	filename = "./data/Pisco.cii.455kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # we know where the source is

	cub = Cube(filename)  # load the map

	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy/beam
	# This map is a single channel collapse over a [CII] emission line, total width of 455 km/s
	# If line fluxes in units of Jy.km/s are preferred, use the lower scaling
	# scale = cub.deltavel()  # channel width in kms, will scale to Jy/beam.km/s (profile) and Jy.km/s (cumulative)

	radius, profile, err = cub.profile_r(ra=ra, dec=dec, maxradius=3, calc_error=True)
	# or alternatively, with same results:
	# radius, flux, err, _ = cub.growing_aperture(ra=ra, dec=dec, maxradius=3, calc_error=True, profile=True)

	print("rms mjy", cub.rms * scale)
	print("err[0]", err[0] * scale)
	fig, ax = plt.subplots(figsize=(4.8, 3))

	col = "navy"
	ax.plot(radius, profile * scale, color=col)
	ax.fill_between(radius, (profile - err) * scale, (profile + err) * scale, color=col, lw=0, alpha=0.2)
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Radius (arcsec)")
	ax.set_ylabel("Azimuthal average (mJy/beam)", color=col)
	ax.tick_params(axis='y', colors=col)
	ax.set_xlim(0, 3)

	radius, flux, err = cub.aperture_r(ra=ra, dec=dec, maxradius=3, calc_error=True)
	# radius, flux, err, _ = cub.growing_aperture(ra=ra, dec=dec, maxradius=3, calc_error=True)  # alternatively

	col = "firebrick"
	ax2 = ax.twinx()
	ax2.plot(radius, flux * scale, color=col)
	ax2.fill_between(radius, (flux - err) * scale, (flux + err) * scale, color=col, lw=0, alpha=0.2)
	ax2.tick_params(direction='in', which="both")
	ax2.set_ylabel("Cumulative flux density (mJy)", color=col)
	ax2.tick_params(axis='y', colors=col)

	plt.savefig("./plots/growing_aperture.pdf", bbox_inches="tight")  # save plot
	plt.show()


def growing_aperture_psf():
	"""
	Compute azimuthally averaged profile and the curve of growth (cumulative flux in aperture) as a function of radius.
	Applied to point spread function (the dirty beam).
	"""

	filename = "./data/Pisco.cii.455kms.psf.fits"
	scale = 1  # no need to scale units, PSF is 1 at its maximum by definition

	cub = Cube(filename)
	radius, profile = cub.profile_r(maxradius=5)  # profile, no coords are given, assume central pixel

	# Compute the beam FWHM. It is elliptical so use geometric mean to obtain a single value for the size.
	mean_beam_fwhm = np.sqrt(cub.beam["bmaj"][0] * cub.beam["bmin"][0])  # [0] is the first (and only) channel
	print("Beam FWHM [arcsec]", mean_beam_fwhm)

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.set_title("Point spread function")

	col = "navy"
	ax.plot(radius, profile * scale, color=col)
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Radius (arcsec)")
	ax.set_ylabel("Azimuthal average", color=col)
	ax.tick_params(axis='y', colors=col)
	ax.set_xlim(0, 5)

	radius, flux = cub.aperture_r(maxradius=5)  # cumulative

	col = "firebrick"
	ax2 = ax.twinx()
	ax2.plot(radius, flux * scale, color=col)
	ax2.tick_params(direction='in', which="both")
	ax2.set_ylabel("Cumulative", color=col)
	ax2.tick_params(axis='y', colors=col)

	plt.savefig("./plots/growing_aperture_psf.pdf", bbox_inches="tight")  # save plot
	plt.show()


def growing_aperture_technical():
	"""
	Compute curve of growths in multiple maps up to some maximum radius.
	Derive corrected flux using residual scaling.
	"""
	filename = "./data/Pisco.cii.455kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # we know where the source is
	scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy/beam

	mcub = MultiCube(filename)  # load maps
	radius, flux, err, tab = mcub.growing_aperture_corrected(ra=ra, dec=dec, maxradius=3, calc_error=True)

	# tab.write("growth.txt", format="ascii.fixed_width", overwrite=True)  # save results in a human readable format

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.set_title("Curves of growth")
	ax.plot(radius, flux * scale, color="firebrick", lw=2, label="Corrected")
	ax.fill_between(radius, (flux - err) * scale, (flux + err) * scale, color="firebrick", lw=0, alpha=0.2)

	ax.plot(radius, tab["flux_dirty"]*scale, label="Dirty", color="black", ls=":")
	ax.plot(radius, tab["flux_clean"]*scale, label="Cleaned components only", ls="-.", color="navy")
	ax.plot(radius, tab["flux_residual"]*scale, label="Residual", ls="--", color="orange")
	ax.plot(radius, tab["flux_image"]*scale, label="Uncorrected: clean + residual", dashes=[10, 3], color="forestgreen")
	ax.plot(radius, tab["epsilon"], color="gray", label="Clean-to-dirty beam ratio")
	ax.axhline(0, color="gray", lw=0.5, ls=":")

	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Radius (arcsec)")
	ax.set_ylabel("Cumulative flux density (mJy)")
	ax.tick_params(direction='in', which="both")
	ax.set_xlim(0,3)

	ax.legend(bbox_to_anchor=(1, 0.8))

	plt.savefig("./plots/growing_aperture_technical.pdf", bbox_inches="tight")  # save plot
	plt.show()


def growing_aperture_paper():
	filename = "./data/Pisco.cii.455kms.image.fits"
	ra, dec = (205.533741, 9.477317341)  # we know where the source is
	redshift = (1900.538 / 222.547) - 1  # redshift from the observed line peak, z= freq_rest_CII / freq_obs - 1
	aper_rad = 1.3  # final manually chosen aperture radius

	mcub = MultiCube(filename)  # load maps

	# This map is a single channel collapse over a [CII] emission line, total width of 455 km/s
	# If line fluxes in units of Jy.km/s are preferred, use the lower scaling
	scale = mcub["image"].deltavel()  # channel width in kms

	radius, flux, err, tab = mcub.growing_aperture_corrected(ra=ra, dec=dec, maxradius=3, calc_error=True)

	# tab.write("growth.txt", format="ascii.fixed_width", overwrite=True)  # save results in a human readable format

	fig, ax = plt.subplots(figsize=(4.8, 3))
	ax.plot(radius, flux * scale, color="firebrick", lw=2, label="Corrected")
	ax.fill_between(radius, (flux - err) * scale, (flux + err) * scale, color="firebrick", lw=0, alpha=0.2)
	ax.plot(radius, tab["flux_image"]*scale, label="Uncorrected", ls="--", color="gray")

	ax.axvline(aper_rad, color="gray", lw=0.75, ls=":", label="Chosen aperture size")

	# Could obtain just the single flux value at given aper_rad with
	# flux, err, tab = mcub.spectrum_corrected(ra=ra, dec=dec, radius=aper_rad, calc_error=True)
	# print(flux*scale,err*scale)
	ax.tick_params(direction='in', which="both")
	ax.set_xlabel("Aperture radius (arcsec)")
	ax.set_ylabel("Line flux density (Jy km/s)")
	ax.set_xlim(0,3)
	ax.legend(loc="lower right", frameon=False)

	# add physical distances scale
	kpc_per_arcsec = iftools.arcsec2kpc(redshift)

	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim()[0] * kpc_per_arcsec, ax.get_xlim()[1] * kpc_per_arcsec)
	ax2.set_xlabel("Radius (kpc)")
	ax2.tick_params(direction='in', which="both")

	plt.savefig("./plots/growing_aperture_paper.pdf", bbox_inches="tight")  # save plot
	plt.show()


def main():
	spectrum_single_pixel()
	spectrum_aperture()
	spectrum_aperture_technical()
	spectrum_aperture_paper()

	growing_aperture()
	growing_aperture_psf()
	growing_aperture_technical()
	growing_aperture_paper()



if __name__ == "__main__":
	main()
