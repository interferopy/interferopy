"""
Examples of interferopy package usage.

From quick look plots, over technical data assessment plots, to paper grade plots.

Author: Mladen Novak, 2020
# Last Contribution: Romain Meyer, Jan 2021
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe
from scipy.optimize import curve_fit
import numpy as np
from astropy.table import Table
import astropy.units as u
from interferopy.cube import Cube, MultiCube
import interferopy.tools as iftools
from scipy.constants import c
# The three libraries below are only needed for wcsaxes plots
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes


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

    #plt.savefig("./plots/spectrum_single_pixel.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/spectrum_single_pixel.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()


def spectrum_aperture():
    """
    Extract aperture spectrum at the specified position.
    """
    filename = "./data/Pisco.cube.50kms.image.fits"
    ra, dec = (205.533741, 9.477317341)  # we know where the source is
    scale = 1e3  # map units are Jy/beam, will use to scale fluxes to mJy

    cub = Cube(filename)
    aper = 1.3
    flux, err = cub.aperture_value(ra=ra, dec=dec, radius=aper, calc_error=True)
    # flux, err _ = cub.spectrum(ra=ra, dec=dec, radius=1.3, calc_error=True)  # alternatively

    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.set_title("Integrated aperture spectrum")
    ax.plot(cub.freqs, flux * scale, color="black", drawstyle='steps-mid', lw=0.75,
            label="Spectrum within r=" + str(aper) + '"')
    ax.fill_between(cub.freqs, flux * scale, 0, color="skyblue", step='mid', lw=0, alpha=0.3)
    ax.plot(cub.freqs, err * scale, color="gray", ls=":", label=r"1$\sigma$ error")  # 1sigma error
    ax.tick_params(direction='in', which="both")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Aperture flux density (mJy)")
    ax.legend(frameon=False)

    #plt.savefig("./plots/spectrum_aperture.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/spectrum_aperture.png", bbox_inches="tight", dpi=72)  # web raster version

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

    txt = "[CII] Flux = " + str(iftools.sigfig(integral_fit, 2)) \
          + r" $\pm$ " + str(iftools.sigfig(integral_err, 1)) + " Jy km/s\n" \
          + "[CII] FWHM = " + str(iftools.sigfig(int(fwhm_kms), 2)) \
          + r" $\pm$ " + str(iftools.sigfig(int(fwhm_err_kms), 1)) + " km/s\n" \
          + "Freq = " + str(iftools.sigfig(nu, 6)) \
          + r" $\pm$ " + str(iftools.sigfig(nu_err, 1)) + " GHz\n" \
          + "Continuum = " + str(iftools.sigfig(cont * scale, 2)) \
          + r" $\pm$ " + str(iftools.sigfig(cont_err * scale, 1)) + " mJy\n"

    # print("Gaussian fit:")
    # print("Flux = " + str(iftools.sigfig(integral_fit, 2)) + " +- " + str(iftools.sigfig(integral_err, 1)) + " Jy.km/s")
    # print("FWHM = " + str(iftools.sigfig(fwhm_kms, 2)) + " +- " + str(iftools.sigfig(fwhm_err_kms, 1)) + " km/s")
    # print("Freq = " + str(iftools.sigfig(nu, 7)) + " +- " + str(iftools.sigfig(nu_err, 1)) + " GHz")

    # plot the spectrum, fill around the fitted continuum value
    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.plot(freqs, spectrum * scale, color="black", drawstyle='steps-mid', lw=0.75)
    ax.fill_between(freqs, spectrum * scale, cont * scale, color="skyblue", step='mid', lw=0, alpha=0.3)

    ax.text(0.98, 0.95, txt, va='top', ha='right', transform=ax.transAxes)

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

    #plt.savefig("./plots/spectrum_aperture_paper.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/spectrum_aperture_paper.png", bbox_inches="tight", dpi=72)  # web raster version

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
    ax2.set_ylabel("Clean-to-dirty\nbeam ratio: " + r"$\epsilon$")
    ax2.set_ylim(-1.5, 1.5)

    #plt.savefig("./plots/spectrum_aperture_technical.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/spectrum_aperture_technical.png", bbox_inches="tight", dpi=72)  # web raster version

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

    #plt.savefig("./plots/growing_aperture.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/growing_aperture.png", bbox_inches="tight", dpi=72)  # web raster version

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

    #plt.savefig("./plots/growing_aperture_psf.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/growing_aperture_psf.png", bbox_inches="tight", dpi=72)  # web raster version

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

    ax.plot(radius, tab["flux_dirty"] * scale, label="Dirty", color="black", ls=":")
    ax.plot(radius, tab["flux_clean"] * scale, label="Cleaned components only", ls="-.", color="navy")
    ax.plot(radius, tab["flux_residual"] * scale, label="Residual", ls="--", color="orange")
    ax.plot(radius, tab["flux_image"] * scale, label="Uncorrected: clean + residual", dashes=[10, 3],
            color="forestgreen")
    ax.plot(radius, tab["epsilon"], color="gray", label="Clean-to-dirty beam ratio")
    ax.axhline(0, color="gray", lw=0.5, ls=":")

    ax.tick_params(direction='in', which="both")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Cumulative flux density (mJy)")
    ax.tick_params(direction='in', which="both")
    ax.set_xlim(0, 3)

    ax.legend(bbox_to_anchor=(1, 0.8))

    #plt.savefig("./plots/growing_aperture_technical.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/growing_aperture_technical.png", bbox_inches="tight", dpi=72)  # web raster version

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
    ax.plot(radius, tab["flux_image"] * scale, label="Uncorrected", ls="--", color="gray")

    ax.axvline(aper_rad, color="gray", lw=0.75, ls=":", label="Chosen aperture size")

    # Could obtain just the single flux value at given aper_rad with
    # flux, err, tab = mcub.spectrum_corrected(ra=ra, dec=dec, radius=aper_rad, calc_error=True)

    # print(flux*scale,err*scale)
    ax.tick_params(direction='in', which="both")
    ax.set_xlabel("Aperture radius (arcsec)")
    ax.set_ylabel("Line flux density (Jy km/s)")
    ax.set_xlim(0, 3)
    ax.legend(loc="lower right", frameon=False)

    # add physical distances scale
    kpc_per_arcsec = iftools.arcsec2kpc(redshift)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * kpc_per_arcsec, ax.get_xlim()[1] * kpc_per_arcsec)
    ax2.set_xlabel("Radius (kpc)")
    ax2.tick_params(direction='in', which="both")

    #plt.savefig("./plots/growing_aperture_paper.pdf", bbox_inches="tight")  # save plot
    plt.savefig("./thumbnails/growing_aperture_paper.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()


def map_single_paper():
    """
    Plot a single 2D map with contours, synthesised beam, colorbar, and text overlay.
    """
    filename = "./data/Pisco.cii.455kms.image.fits"
    ra, dec = (205.533741, 9.477317341)  # we know where the source is
    cutout = 3  # radius of the cutout in arcsec (full panel is 2xcutout)
    aper_rad = 1.3
    titletext = r"[CII] 158 $\mu$m"

    cub = Cube(filename)  # load map

    scale = 1e3  # Jy/beam to mJy/beam
    # scale = cub.deltavel()  # use this to scale units to Jy/beam km/s

    fig = plt.figure(figsize=(3, 3))  # nrows_ncols=(2,4)

    # Use the ImageGrid to display a map and a colorbar to the right
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.05, share_all=True,
                     cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.05)
    ax = grid[0]

    # create a smaller cutout for plotting, a subimage
    # Note: there should be a better way to plot cutouts, but this works.
    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))  # slightly larger cutout than required for edge bleeding
    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    # get the cutout; warning: no index out of bounds checking is done here
    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale  # scale units

    # for color scaling
    vmax = np.nanmax(subim)
    vmin = -0.1 * vmax

    # show image
    axim = ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)

    # calc rms and plot contours
    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-8, -4, -2]) * rms, zorder=1, linewidths=0.5,
               linestyles="--")
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5,
               linestyles="-")

    # add beam, angle is between north celestial pole and major axis, angle increases toward increasing RA
    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8),
                      width=cub.beam["bmin"], height=cub.beam["bmaj"], angle=-cub.beam["bpa"],
                      edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    # add circular aperture
    ellipse = Ellipse(xy=(0, 0), width=2 * aper_rad, height=2 * aper_rad, angle=0,
                      edgecolor='white', fc="none", lw=1, ls=":")
    ax.add_patch(ellipse)

    # add text on top of the map
    ax.text(0.5, 0.95, titletext,
            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],
            va='top', ha='center', color="white", transform=ax.transAxes)

    # add colorbar
    cb = ax.cax.colorbar(axim)
    cb.set_label_text(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")

    #plt.savefig("./plots/map_single_paper.pdf", bbox_inches="tight", dpi=600)  # need higher dpi for crisp data pixels
    plt.savefig("./thumbnails/map_single_paper.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()


def map_channels_paper():
    """
    Plot a channel maps from the cube.
    """
    filename = "./data/Pisco.cube.50kms.image.fits"
    ra, dec, freq = (205.533741, 9.477317341, 222.547)  # we know where the source is
    cutout = 1.5  # arcsec
    scale = 1e3  # Jy/beam to mJy/beam

    # set up the channel map grid (change the figure size if necessary for font scaling)
    nrows = 3
    ncols = 3
    figsize = (6, 6)
    idx_center = int(0.5 * nrows * ncols)  # index of the central panel

    cub = Cube(filename)  # load map

    ch_peak = cub.freq2pix(freq)  # referent channel in the cube - one with the peak line emission

    # velocity offset of each channel from the referent frequency
    velocities = cub.vels(freq)  # use the frequency from the spectral fit
    # velocities = cub.vels(cub.freqs[ch_peak])  # it's nicer if the ch_peak velocity is set to exactly 0 km/s.

    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05, share_all=True,
                     cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.05)

    # get the extent of the cutouts
    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))  # slightly larger cutout than required, for edge bleeding
    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    # use the reference channel to set the colorbar scale
    # vmax = np.nanmax(cub.im[px - r:px + r + 1, py - r:py + r + 1, ch_peak]) * scale
    # or use a range of channels to find the max value for colorbar scaling:
    vmax = np.nanmax(cub.im[px - r:px + r + 1, py - r:py + r + 1,
                     ch_peak - idx_center:ch_peak - idx_center + nrows * ncols]) * scale
    vmin = -0.1 * vmax

    for i in range(nrows * ncols):
        ax = grid[i]
        ch = ch_peak - idx_center + i  # this will put the peak channel on i = idx_center position
        subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, ch] * scale  # scale units

        # show image
        axim = ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)

        # set limits to exact cutout size
        ax.set_xlim(cutout, -cutout)
        ax.set_ylim(-cutout, cutout)

        # calc rms and plot contours
        rms = cub.rms[ch] * scale
        ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-8, -4, -2]) * rms, zorder=1,
                   linewidths=0.5, linestyles="--")
        ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 8, 16, 32]) * rms, zorder=1,
                   linewidths=0.5, linestyles="-")

        # add beam, angle is between north celestial pole and major axis, angle increases toward increasing RA
        ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8),
                          width=cub.beam["bmin"][ch], height=cub.beam["bmaj"][ch], angle=-cub.beam["bpa"][ch],
                          edgecolor='black', fc='w', lw=0.75)
        ax.add_patch(ellipse)

        # add circular aperture to the central panel
        if i == idx_center:
            aper_rad = 1.3
            ellipse = Ellipse(xy=(0, 0), width=2 * aper_rad, height=2 * aper_rad, angle=0,
                              edgecolor='white', fc="none", lw=1, ls=":")
            ax.add_patch(ellipse)

        # add text on top of the map
        # paneltext = str(cub.freqs[ch])
        paneltext = str(int(velocities[ch])) + " km/s"
        ax.text(0.5, 0.95, paneltext,
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()],
                va='top', ha='center', color="white", transform=ax.transAxes)

        ax.tick_params(direction='in', which="both")

        # Could put global labels to the figure, but in this case just put labels to the middle edge panels
        if i == (ncols * int(nrows / 2)):
            ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
        if i == (nrows * ncols - int(ncols / 2) - 1):
            ax.set_xlabel(r"$\Delta$ RA (arcsec)")

    # add colorbar
    cb = ax.cax.colorbar(axim)
    cb.set_label_text(r"$S_\nu$ (mJy beam$^{-1}$)")

    #plt.savefig("./plots/map_channels_paper.pdf", bbox_inches="tight", dpi=600)  # need higher dpi for crisp data pixels
    plt.savefig("./thumbnails/map_channels_paper.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()

    return None


def map_technical():
    """
    Plot several maps generated in the cleaning process (CASA tclean outputs).
    """
    filename = "./data/Pisco.cii.455kms.image.fits"
    ra, dec, freq = (205.533741, 9.477317341, 222.547)  # we know where the source is
    cutout = 2.5  # arcsec, check that it is smaller than the image!
    # scale = 1e3  # Jy/beam to mJy/beam

    ch = 0  # channel to plot (for simple 2D maps, the first channel is the only channel)

    mcub = MultiCube(filename)
    mcub.make_clean_comp()  # generate clean component map

    fig, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=3, sharex=True, sharey=True)

    # fig = plt.figure(figsize=figsize)
    # grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05, share_all=True,
    # 				 cbar_location="right", cbar_mode="single", cbar_size="3%", cbar_pad=0.05)

    # save a reference to the main image for easier use
    cub = mcub["image"]
    # get the extent of the cutouts
    px, py = cub.radec2pix()  # do not give coordinates, take the central pixel
    r = int(np.round(cutout * 1.05 / cub.pixsize))  # slightly larger cutout than required for edge bleeding
    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    ra, dec = cub.pix2radec(px, py)
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    # Image map
    ax = axes[0, 0]
    subim = mcub["image"].im[px - r:px + r + 1, py - r:py + r + 1, ch]  # scale units
    vmax = np.nanmax(subim)
    vmin = -0.1 * vmax
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title("Cleaned")

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    # calc rms and plot contours
    # rms = mcub["image"].rms[ch]
    # ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-8, -4, -2]) * rms, zorder=1,
    # 		   linewidths=0.5, linestyles="--")
    # ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 8, 16, 32]) * rms, zorder=1,
    # 		   linewidths=0.5, linestyles="-")

    # add beam, angle is between north celestial pole and major axis, angle increases toward increasing RA
    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8),
                      width=cub.beam["bmin"], height=cub.beam["bmaj"], angle=-cub.beam["bpa"],
                      edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # Dirty map
    ax = axes[0, 1]
    subim = mcub["dirty"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title("Dirty")

    # Residual map
    ax = axes[0, 2]
    subim = mcub["residual"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title("Residual")

    # Clean components map
    ax = axes[1, 0]
    subim = mcub["clean.comp"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    # Used generated map "clean.comp", alternatively, plot the difference directly
    # subim = (mcub["image"].im - mcub["residual"].im)[px - r:px + r + 1, py - r:py + r + 1, ch]
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title("Clean component")

    # model
    ax = axes[1, 1]
    subim = mcub["model"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    # model has units of Jy/pixel so generate different maximums here
    vmax = np.nanmax(subim)
    vmin = -0.1 * vmax
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title("Model")

    # PSF
    ax = axes[1, 2]
    subim = mcub["psf"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=-0.05, vmax=0.5, extent=extent)
    ax.set_title("PSF")

    # PB
    # Not needed for targeted obs where the source is in the phase center (PB = 1)
    # ax = axes[1, 2]
    # subim = mcub["pb"].im[px - r:px + r + 1, py - r:py + r + 1, ch]
    # ax.imshow(subim.T, origin='lower', cmap="RdBu_r", vmin=0.95, vmax=1, extent=extent)
    # ax.set_title("PB")

    #plt.savefig("./plots/map_technical.pdf", bbox_inches="tight", dpi=600)  # need higher dpi for crisp data pixels
    plt.savefig("./thumbnails/map_technical.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()


def map_wcsaxes():
    """
    Plot a map with wcsaxes and semi-logarithmic scaling. Scaling can be controlled to ensure consistency across several
    plots. Prints fluxes of the central source.
    """

    # Two parameters that control the scaling to ensure consistency across plots. If none, uses maximum of map and
    # 5sigma rms noise aas threshold to go from linear to logarithmmic
    vmax = None
    linthres = None

    filename = "./data/Pisco.cii.455kms.image.fits"
    cub = Cube(filename)

    # we know where the source is
    cutout = 4  # radius of the cutout in arcsec (full panel is 2xcutout)
    cutout_pix = cutout / cub.pixsize  # DEC radius of cutout in arcsec
    aper_rad = 2

    scale = 1e3  # Jy/beam to mJy/beam

    fig = plt.figure(figsize=(3, 3))

    # Use the ImageGrid to display a map and a colorbar to the right
    ax = plt.subplot(projection=cub.wcs, label='overlays', slices=('x', 'y', 200))
    lon = ax.coords[0]
    lat = ax.coords[1]

    # for color scaling
    if vmax == None:
        vmax = np.nanmax(cub.im[:, :, 0] * scale)
        linthres = 5 * np.std(cub.im[:, :, 0].T * scale)

    # show image
    axim = ax.imshow(cub.im[:, :, 0].T * scale, origin='lower', cmap="PuOr_r", vmin=-vmax, vmax=vmax, zorder=-1,
                     norm=colors.SymLogNorm(linthresh=linthres,
                                            linscale=0.5,
                                            vmin=-vmax, vmax=vmax))

    # calc rms and plot contours
    rms = cub.rms[0] * scale

    ax.contour(cub.im[:, :, 0].T * scale, colors="gray", levels=np.array([-8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")
    ax.contour(cub.im[:, :, 0].T * scale, colors="black", levels=np.array([2, 4, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")

    # add beam, angle is between north celestial pole and major axis, angle increases toward increasing RA
    ellipse = Ellipse(xy=(cub.im.shape[0] / 2 - cutout_pix * 0.75, cub.im.shape[0] / 2 - cutout_pix * 0.75),
                      width=cub.beam["bmin"] / cub.pixsize, height=cub.beam["bmaj"] / cub.pixsize,
                      angle=cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cub.im.shape[0] / 2 - cutout_pix, cub.im.shape[0] / 2 + cutout_pix)
    ax.set_ylim(cub.im.shape[1] / 2 - cutout_pix, cub.im.shape[1] / 2 + cutout_pix)

    # add circular aperture
    ellipse = Ellipse(xy=(cub.im.shape[1] / 2, cub.im.shape[1] / 2), width=2 * aper_rad / cub.pixsize,
                      height=2 * aper_rad / cub.pixsize, angle=0,
                      edgecolor='maroon', fc="none", lw=1, ls=":")
    ax.add_patch(ellipse)

    # add text on top of the map
    ax.text(0.5, 0.95, r'[CII] 158 $\mu$m',
            path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],
            va='top', ha='center', color="white", transform=ax.transAxes)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", axes_class=maxes.Axes)
    cbar = fig.colorbar(axim, cax=cax, orientation='horizontal',
                        ticks=[-int(vmax), -2 * linthres, -linthres / 2, linthres / 2, 2 * linthres, int(vmax)],
                        format='%0.1f')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_text(r"$S_\nu$ (mJy beam$^{-1}$)")
    cax.xaxis.set_label_position('top')

    lon.set_ticks(number=4)
    lat.set_minor_frequency(1)
    ax.tick_params(direction='in', which="both")
    ax.set_xlabel(r"RA ")
    ax.set_ylabel(r"Dec")

    #plt.savefig("./plots/map_wcsaxes.pdf", bbox_inches="tight", dpi=600)  # need higher dpi for crisp data pixels
    plt.savefig("./thumbnails/map_wcsaxes.png", bbox_inches="tight", dpi=72)  # web raster version

    plt.show()

    return vmax, linthres


def map_channels_wcsaxes():
    """
    Plot a channel maps from the cube using wcsaxes and lin-log scaling. Disclaimer: ccolorbar size is optimized for 3x3
    grid.
    """
    filename = "./data/Pisco.cube.50kms.image.fits"

    cutout = 2  # arcsec
    scale = 1e3  # Jy/beam to mJy/beam

    ra, dec, freq = (205.533741, 9.477317341, 222.547)

    # set up the channel map grid (change the figure size if necessary for font scaling)
    nrows = 3
    ncols = 3
    figsize = (8, 8)
    idx_center = int(0.5 * nrows * ncols)  # index of the central panel

    # if needed, set offset_v to make channel maps blue- or red-wards of the line, but keeping the line as the
    # reference to write in the title of each channel (e.g. \pm Delta_v = XX km/s)
    offset_v = 0

    cub = Cube(filename)  # load map
    cutout_pix = cutout / cub.pixsize
    ch_peak = cub.freq2pix(
        freq * (1 + offset_v / 3e5))  # referent channel in the cube - one with the peak line emission

    # velocity offset of each channel from the referent frequency
    # velocities = cub.vels(freq)  # use the frequency from the spectral fit
    velocities = cub.vels(cub.freqs[ch_peak])  # it's nicer if the ch_peak velocity is set to exactly 0 km/s.

    fig = plt.figure(figsize=figsize)

    # get the extent of the cutouts
    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))  # slightly larger cutout than required, for edge bleeding

    # use the reference channel to set the colorbar scale
    # vmax = np.nanmax(cub.im[px - r:px + r + 1, py - r:py + r + 1, ch_peak]) * scale
    # or use a range of channels to find the max value for colorbar scaling:
    vmax = np.nanmax(cub.im[px - r:px + r + 1, py - r:py + r + 1,
                     ch_peak - idx_center:ch_peak - idx_center + nrows * ncols]) * scale
    linthres = 3 * np.std(cub.im[px - r:px + r + 1, py - r:py + r + 1,
                          ch_peak - idx_center:ch_peak - idx_center + nrows * ncols]) * scale

    ax_list = []

    for i in range(nrows * ncols):
        ax = fig.add_subplot(str(nrows) + str(ncols) + str(i + 1), projection=cub.wcs, label='overlays',
                             slices=('x', 'y', 50))
        ax_list.append(ax)
        ra = ax.coords[0]
        dec = ax.coords[1]

        ch = ch_peak - idx_center + i  # this will put the peak channel on i = idx_center position
        subim = cub.im[:, :, ch] * scale  # scale units

        # show image
        axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", vmin=-vmax, vmax=vmax, zorder=-1,
                         norm=colors.SymLogNorm(linthresh=linthres,
                                                linscale=0.5,
                                                vmin=-vmax, vmax=vmax))

        # set limits to exact cutout size
        ax.set_xlim(cub.im.shape[0] / 2 - cutout_pix, cub.im.shape[0] / 2 + cutout_pix)
        ax.set_ylim(cub.im.shape[1] / 2 - cutout_pix, cub.im.shape[1] / 2 + cutout_pix)

        # calc rms and plot contours
        rms = cub.rms[ch] * scale
        ax.contour(subim.T, colors="gray", levels=np.array([-8, -4, -2]) * rms, zorder=1,
                   linewidths=0.5, linestyles="--")
        ax.contour(subim.T, colors="black", levels=np.array([2, 4, 8, 16, 32]) * rms, zorder=1,
                   linewidths=0.5, linestyles="-")

        # add beam, angle is between north celestial pole and major axis, angle increases toward increasing RA
        ellipse = Ellipse(xy=(cub.im.shape[0] / 2 - cutout_pix * 0.75, cub.im.shape[1] / 2 - cutout_pix * 0.75),
                          width=cub.beam["bmin"][ch] / cub.pixsize, height=cub.beam["bmaj"][ch] / cub.pixsize,
                          angle=cub.beam["bpa"][ch], edgecolor='black', fc='w', lw=0.75)
        ax.add_patch(ellipse)

        # add circular aperture to the central panel
        if i == idx_center:
            aper_rad = 1.3
            ellipse = Ellipse(xy=(cub.im.shape[0] / 2, cub.im.shape[1] / 2),
                              width=2 * aper_rad / cub.pixsize, height=2 * aper_rad / cub.pixsize, angle=0,
                              edgecolor='firebrick', fc="none", lw=1, ls=":")
            ax.add_patch(ellipse)

        # add text on top of the map
        paneltext = str(int(velocities[ch] + offset_v)) + " km/s"
        ax.text(0.5, 0.95, paneltext,
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()],
                va='top', ha='center', color="white", transform=ax.transAxes)

        ax.tick_params(direction='in', which="both")

        dec.set_axislabel(' ')
        ra.set_axislabel(' ')
        ra.set_ticklabel_visible(False)
        dec.set_ticklabel_visible(False)

        # writing RA and DEC and ticklabels only on outer edges of the grid
        if (i) % nrows == 0:
            dec.set_ticklabel_visible(True)
            dec.set_ticks(number=5)
        if i >= (nrows - 1) * ncols:
            ra.set_ticklabel_visible(True)
            ra.set_ticks(number=2)
        if i == nrows * ncols - 2:
            ra.set_axislabel("RA", fontsize=14)
        if i == int(nrows / 2) * ncols:
            dec.set_axislabel("DEC", fontsize=14)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.11, 0.03, 0.77])
    fig.colorbar(axim, cax=cbar_ax)
    cbar_ax.yaxis.set_label_text(r"$S_\nu$ (mJy beam$^{-1}$)", fontsize=14)
    cbar_ax.yaxis.set_label_position('right')

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    #plt.savefig("./plots/map_channels_wcsaxes.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("./thumbnails/map_channels_wcsaxes.png", bbox_inches="tight", dpi=72)
    plt.show()

    return None


def dust_cont_fit():
    """
    Fit a modified black body dust continuum emission to observed flux density data points.
    File for input flux densities, redshift, and exact fitting method are currently hardcoded.
    Uses dust_sobs, which takes cmb heating and contrast into account by default.

    :return:
    """

    # Points to fit, inputs need so to be in SI units
    t = Table.read("./data/Pisco_continuum_fluxes.txt", format="ascii.commented_header")
    freqs = t["freq_ghz"] * 1e9  # to Hz
    fluxes = t["flux_jy"] * 1e-26  # to W/Hz/m2
    fluxes_err = t["flux_err_jy"] * 1e-26  # to W/Hz/m2

    # Parameters for the black body emission
    # These are either fixed, or used as intial guess for fitting if chosen to be free parameters
    z = 7.5413  # redshift
    dust_mass = 1e37  # in kilograms
    dust_temp = 47  # T_dust in Kelvins
    dust_beta = 1.9  # modified black body exponent

    dust_mass_err = 0
    dust_temp_err = 0
    dust_beta_err = 0

    # several fitting scenarios, choose one
    # lambdas are used to set non-free parameters, because curve_fit wants only the variable and free params
    # Parameters are degenerate, so be careful with error interpretation

    # Fit Mdust only - this could be used if you only have a single point, for example
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass:
                               iftools.dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass), sigma=fluxes_err, absolute_sigma=True)
        dust_mass = popt[0]
        dust_mass_err = np.diagonal(pcov)[0]

    # Fit Mdust and T - to constrain the temperature, the black body peak needs to be sampled
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_temp:
                               iftools.dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_temp), sigma=fluxes_err, absolute_sigma=True)
        dust_mass, dust_temp = popt
        dust_mass_err, dust_temp_err = np.sqrt(np.diagonal(pcov))

    # Fit Mdust and beta - the best option on the Rayleigh-Jeans tail
    if 1:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_beta:
                               iftools.dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_beta), sigma=fluxes_err, absolute_sigma=True)
        dust_mass, dust_beta = popt
        dust_mass_err, dust_beta_err = np.sqrt(np.diagonal(pcov))

    # Fit Mdust and T and beta - not recommended due to degeneracy
    if 0:
        popt, pcov = curve_fit(lambda freqs, dust_mass, dust_temp, dust_beta:
                               iftools.dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                               freqs, fluxes, p0=(dust_mass, dust_temp, dust_beta), sigma=fluxes_err,
                               absolute_sigma=True)
        dust_mass, dust_temp, dust_beta = popt
        dust_mass_err, dust_temp_err, dust_beta_err = np.sqrt(np.diagonal(pcov))

    print("dust_mass (10^8 Msol) = ", iftools.sigfig(dust_mass * u.kg.to(u.solMass) * 1e-8, 3),
          " +- ", iftools.sigfig(dust_mass_err * u.kg.to(u.solMass) * 1e-8, 1))
    print("dust_temp (K) = ", iftools.sigfig(dust_temp, 3), " +- ", iftools.sigfig(dust_temp_err, 1))
    print("dust_beta = ", iftools.sigfig(dust_beta, 3), " +- ", iftools.sigfig(dust_beta_err, 1))

    # integrate the dust SED, get LTIR, LFIR and SFR

    lum_fir, lum_tir, sfr_K98, sfr_k12 = iftools.dust_cont_integrate(dust_mass=dust_mass, dust_temp=dust_temp,
                                                                     dust_beta=dust_beta, print_to_console=True)

    return dust_mass, dust_temp, dust_beta, lum_fir, lum_tir, sfr_K98, sfr_k12


def dust_cont_plot(dust_mass, dust_temp, dust_beta):
    """
    Plot the data points and the fitted dust continuum model. Example with Pisco continuum values.

    Input points file is hardcoded.
    :param dust_mass: in kg
    :param dust_temp: in K
    :param dust_beta: dimensionless
    :return:
    """

    z= 7.5413
    # Points to plot
    t = Table.read("./data/Pisco_continuum_fluxes.txt", format="ascii.commented_header")
    freqs = t["freq_ghz"]
    fluxes = t["flux_jy"] * 1e3  # plot mJy
    fluxes_err = t["flux_err_jy"] * 1e3  # plot mJy

    fig, ax = plt.subplots(figsize=(4.8, 3))

    ax.errorbar(freqs, fluxes, marker="o", color="black", linestyle="", markerfacecolor='skyblue', ms=6,
                label='Measurements')
    ax.fill_between(freqs, fluxes - fluxes_err, fluxes + fluxes_err, color="skyblue", alpha=0.5)  # , label=r"aper err"

    # plot fitted model
    xxx = np.linspace(np.min(freqs) / 5, np.max(freqs) * 5, 1000)  # plot model over this range (GHz)
    ax.plot(xxx, iftools.dust_sobs(xxx * 1e9, z, dust_mass, dust_temp, dust_beta) * 1e26 * 1e3,  # x in GHz, y in mJy
            color="black", linestyle="-", lw=0.75, label="Fitted black body")

    # fill in the FIR region
    xxx = np.linspace(c / (122.5e-6) * 1e-9 / (1 + z), c / (42.5e-6) * 1e-9 / (1 + z), 1000)
    ax.fill_between(xxx, 0, iftools.dust_sobs(xxx * 1e9, z, dust_mass, dust_temp, dust_beta) * 1e26 * 1e3, color="orange",
                    alpha=0.2, label=r"FIR: rest-frame 42.5 - 122.5 $\mu$m")

    # axes details
    ax.set_xlabel(r"$\nu_{obs}$ (GHz)")
    ax.set_ylabel(r"$S_\nu$ (mJy)")  # observed flux densities
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1.4)
    ax.tick_params(direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # add a second axis to display wavelengths in mm
    # wavelength scales as 1/freq, so I manually put ticks to be sure they are in the right spot
    label_mm = np.array([0.2, 0.3, 0.4, 0.5, 1, 2, 5])  # plot these mm ticks
    label_mm_corresponding_ghz = c / (label_mm * 1e-3) * 1e-9  # get at which frequencies do these ticks occur
    label_mm_format = [x if x < 1 else int(x) for x in label_mm]  # do not write decimal part for ticks > 1

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # match the range to the lower axis (make sure manual ticks are within this range)
    ax2.set_xticks(label_mm_corresponding_ghz)
    ax2.set_xticklabels(label_mm_format)  # override tick labels to mm
    ax2.set_xlabel(r"$\lambda_{obs}$ (mm)")
    ax2.tick_params(direction='in')

    #plt.savefig("./plots/dust_continuum_fluxes.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("./thumbnails/dust_continuum_fluxes.png", bbox_inches="tight", dpi=72)
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

    map_single_paper()
    map_channels_paper()
    map_technical()

    map_wcsaxes()
    map_channels_wcsaxes()

    dust_mass, dust_temp, dust_beta, _, _, _, _ =  dust_cont_fit()
    dust_cont_plot(dust_mass=dust_mass, dust_temp=dust_temp, dust_beta=dust_beta)

if __name__ == "__main__":
    main()
