import os
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.coordinates import SkyCoord
from scipy.special import erf, erfinv
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.constants import c


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

    if len(a) < 1:
        # no finite pixels
        rms = np.nan

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
        temp_dustz = (temp_dust ** (4 + beta) + temp_cmb0 ** (4 + beta) * ((1 + z) ** (4 + beta) - 1)) ** (
                1 / (4 + beta))
    else:
        temp_dustz = temp_dust

    if cmb_contrast:
        f_cmb = 1. - blackbody(nu_rest, temp_cmb) / blackbody(nu_rest, temp_dustz)
    else:
        f_cmb = 1

    flux_obs = f_cmb * (1 + z) / (4 * np.pi * dl ** 2) * dust_lum(nu_rest, mass_dust, temp_dustz, beta)

    return flux_obs

def dust_cont_integrate(dust_mass, dust_temp, dust_beta, print_to_console=False):
    """
    Integrate over the IR spectral energy distribution. Calculate SFR based on Kennicut relation.
    Prints output if print parameter==True and returns values.
    :param dust_mass: in kg
    :param dust_temp: in K
    :param dust_beta: dimensionless
    :param print_to_console: print values to console
    :return: lum_tir (8-1000 microns), lum_fir (42.5-122.5 microns) in solLum , SFR-K98, SFR-K12 in Msun/yr
    """

    # Total IR is 8 - 1000 microns
    lum_tir = integrate.quad(lambda x: dust_lum(x, dust_mass, dust_temp, dust_beta), c / (1000e-6), c / (8e-6))

    # Far IR is 42.5 - 122.5 microns
    lum_fir = integrate.quad(lambda x: dust_lum(x, dust_mass, dust_temp, dust_beta), c / (122.5e-6), c / (42.5e-6))

    # Kennicutt+98 relation
    sfr_Kennicutt98 = lum_tir[0] * u.W.to(u.solLum) * 1e-10

    # Kennicutt+12 relation scaled to Chabrier IMF
    sfr_Kennicutt12 = 10 ** (np.log10(lum_tir[0] * u.W.to(u.erg / u.s)) - 43.41) / 1.7

    if print_to_console:
        print("Ltir (10^12 Lsol) = ", sigfig(lum_tir[0] * u.W.to(u.solLum) * 1e-12, 3))
        print("Lfir (10^12 Lsol) =", sigfig(lum_fir[0] * u.W.to(u.solLum) * 1e-12, 3))
        print("SFR_Kennicutt98 (Msol/yr)", sigfig(sfr_Kennicutt98, 3))
        print("SFR_Kennicutt12 (Msol/yr)", sigfig(sfr_Kennicutt12, 3))

    return lum_tir,lum_fir, sfr_Kennicutt98,sfr_Kennicutt12


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


def line_stats_sextractor(catalogue, binning, SNR_min=5):
    '''
    A convenience function extracting high-SNR clumps from a FINDCLUMP(s) output catalogue, adding the binning and
    changing the frequency to GHz, plus reordering for output catalogues

    :param catalogue: findclump output
    :param binning: kernel half-width = (minwidth -1)/2
    :param SNR_min: mininum SNR above which to pick clumps
    :return: reordered, cleaned catalogue
    '''

    # By definition, the output of findclumps is
    # SNR_WIN FLUX_MAX X_IMAGE Y_IMAGE RA(J2000,deg) DEC(J2000,deg) Center_channel RMS
    snr = catalogue[:, 1] / catalogue[:, 7]
    freq_GHz = catalogue[:, 8]  # catalogue from interferopy.Cube is already in GHz!

    ind_high_SN = np.where(snr > SNR_min)[0]

    # writing RA DEC FREQ(GHZ) X Y SNR FLUX_MAX BINNING
    shuffled_catalogue = np.zeros(shape=(len(ind_high_SN), 8))
    shuffled_catalogue[:, 0] = catalogue[ind_high_SN, 4]
    shuffled_catalogue[:, 1] = catalogue[ind_high_SN, 5]
    shuffled_catalogue[:, 2] = freq_GHz[ind_high_SN]
    shuffled_catalogue[:, 3] = catalogue[ind_high_SN, 2]
    shuffled_catalogue[:, 4] = catalogue[ind_high_SN, 3]
    shuffled_catalogue[:, 5] = snr[ind_high_SN]
    shuffled_catalogue[:, 6] = catalogue[ind_high_SN, 1]
    shuffled_catalogue[:, 7] = np.ones(len(ind_high_SN)) * binning

    return shuffled_catalogue


def run_line_stats_sex(sextractor_catalogue_name,
                       binning_array=np.arange(1, 20, 2),
                       SNR_min=3):
    '''
    Merges, cleans and reformat clump catalogues (positive or negative) of different kernel widths. Also makes a DS9
    region file for all clumps above a chosen threshold. Made to operate over all kernel half-widths for convenience.

    :param sextractor_catalogue_name: Generic catalogue name for the field, excluding kernel half-width
    :param binning_array: array of kernel half-width to process for the given field
    :param SNR_min: Threshold SN to select clump detections
    :return: Saves region files (input name +".reg") and catalogues (input name +".cat") for later study
    '''

    # open region files in overwrite mode
    with open(sextractor_catalogue_name+'_minSNR_'+str(SNR_min)+'.reg', 'w+') as clumps_reg:
        clumps_reg.write('# Region file format: DS9 version 4.1 \n '
                         'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 '
                         'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
        clumps_reg.write('fk5 \n')

        clumps_name_out = sextractor_catalogue_name+'_minSNR_'+str(SNR_min)+'.cat'

        for binning in binning_array:
            catalogue = np.loadtxt(sextractor_catalogue_name + '_kw' + str(int(binning)) + '.cat')

            cat_for_out_file = line_stats_sextractor(catalogue=catalogue,
                                                     binning=binning,
                                                     SNR_min=SNR_min)

            for x in cat_for_out_file:
                clumps_reg.write('circle(' + '{:9.5f}'.format(x[0]) + ',' + '{:9.5f}'.format(x[1])
                                 + ',0.5") \n  # text(' + '{:9.5f}'.format(x[0]) + ',' + '{:9.5f}'.format(x[1])
                                 + ')   text={' + '{:7.3f}'.format(x[2]) + '} \n')

            if binning == binning_array[0]:
                np.savetxt(fname=clumps_name_out, X=cat_for_out_file,
                           fmt=['%9.5f', '%9.5f', '%8.4f', '%5.1f', '%5.1f', '%6.2f', '%9.6f', '%2.0f'],
                           header="RA DEC FREQ_GHZ X Y SNR FLUX_MAX BINNING")
            else:
                with open(clumps_name_out, "ab") as f:
                    np.savetxt(fname=f, X=cat_for_out_file,
                               fmt=['%9.5f', '%9.5f', '%8.4f', '%5.1f', '%5.1f', '%6.2f', '%9.6f', '%2.0f'],
                               header="RA DEC FREQ_GHZ X Y SNR FLUX_MAX BINNING")


def crop_doubles(cat_name, delta_offset_arcsec=2, delta_freq=0.1, verbose=False):
    '''
    Takes a catalogue of clumps and group sources likely from the same target. Tolerance in sky posoition and frequency
    to be given. If the data is not continuum-subtracted, continuum sources will result in multiple groups of clumps
    separated by delta_freq. Best used on continuum-subtracted data.

    :param cat_name:
    :param delta_offset_arcsec:
    :param delta_freq:
    :param verbose: show a progress bar, useful for time-consuming crops
    :return: Writes a copy of the input catalogue, with the added group number for each clump ("_groups.cat"),
        a reduced catalogue with the highest SN detection for each group ("_cropped.cat"), a region files to plot the
        positions of the latter.
    '''

    delta_offset_deg = delta_offset_arcsec / 3600.

    catalogue_data = np.loadtxt(cat_name)

    # sort by SNR decreasing
    catalogue_data = catalogue_data[catalogue_data[:, 5].argsort()[::-1]]

    cosd_array = np.cos(catalogue_data[:, 1] * np.pi / 180.)

    group = -np.ones(len(catalogue_data))
    ncnt = 0

    if verbose:
        print("Cropping doubles:")
        iterable = tqdm(range(len(catalogue_data)))
    else:
        iterable = range(len(catalogue_data))
    for i in iterable:

        if group[i] < 0:
            group[i] = ncnt
            ncnt += 1

            ra = catalogue_data[i, 0]
            dec = catalogue_data[i, 1]
            cosd = cosd_array[i]
            freq = catalogue_data[i, 2]
            distances = np.sqrt(((ra - catalogue_data[:, 0]) * cosd) ** 2 + (dec - catalogue_data[:, 1]) ** 2)
            matches = (distances < delta_offset_deg) & (np.abs(freq - catalogue_data[:, 2]) < delta_freq) & (group < 0)

            if np.sum(matches) > 0:
                ind_same_group = np.where(matches)[0]
                group[ind_same_group] = group[i]

    catalogue_final = np.hstack([catalogue_data, np.reshape(group, (len(group), 1))])

    catalogue_final = catalogue_final[catalogue_final[:, -1].argsort()]

    np.savetxt(cat_name[:-4] + '_groups.cat', catalogue_final,
               fmt=['%9.5f', '%9.5f', '%8.4f', '%5.1f', '%5.1f', '%6.2f', '%9.6f', '%2.0f', '%2.0i'],
               header="RA DEC FREQ_GHZ X Y SNR FLUX_MAX BINNING GROUP")

    catalogue_cropped_best = np.zeros((ncnt, len(catalogue_final[0])))
    if verbose:
        print("Filtering highest S/N sources:")
        iterable = tqdm(range(ncnt))
    else:
        iterable = range(ncnt)
    for i in iterable:
        ind = np.where([catalogue_final[:, -1] == i])[1]
        if len(ind) > 1:
            argmax_SN = np.argmax(catalogue_final[ind, 5])

            catalogue_cropped_best[i, :] = catalogue_final[ind][argmax_SN]
        else:
            catalogue_cropped_best[i, :] = catalogue_final[ind, :]

    np.savetxt(cat_name[:-4] + '_cropped.cat', catalogue_cropped_best,
               fmt=['%9.5f', '%9.5f', '%8.4f', '%5.1f', '%5.1f', '%6.2f', '%9.6f', '%2.0f', '%2.0i'],
               header="RA DEC FREQ_GHZ X Y SNR FLUX_MAX BINNING GROUP")

    catalogue_cropped_best_region = open(cat_name[:-4] + '_cropped.reg', 'w+')
    catalogue_cropped_best_region.write('# Region file format: DS9 version 4.1 \n '
                                        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 '
                                        'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
    catalogue_cropped_best_region.write('fk5 \n')
    for x in catalogue_cropped_best:
        catalogue_cropped_best_region.write('circle(' + '{:9.5f}'.format(x[0]) + ',' + '{:9.5f}'.format(x[1])
                                            + ',0.5") \n  # text(' + '{:9.5f}'.format(x[0]) + ',' + '{:9.5f}'.format(x[1])
                                            + ')   text={' + '{:7.3f}'.format(x[2]) + '}\n')


def fidelity_function(sn, sigma, c):
    return 0.5 * erf((sn - c) / sigma) + 0.5


def fidelity_selection(cat_negative, cat_positive, bin_edges=np.arange(0, 20, 0.1),
                       i_SN=5, fidelity_threshold=0.6, min_SN_fit=4.0, verbose=True):
    '''
    Fidelity selection following Walter et al. 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...833...67W/abstract) to
    select clumps which are more likely to be positive than negative. Plot the selection and threshold if required.

    :param cat_negative: Catalogue of negative clump detections
    :param cat_positive: Catalogue of positive clump detections
    :param bin_edges: SN bins for the fidelity analysis
    :param i_SN: index of the SNR in the catalogues (if catalogues were produced by the internal interferopy findclumps function i_SN =5)
    :param plot_name: if different than "" plot the fidelity function and threshold and save to given name
    :param min_SN_fit: minimum SN for fidelity fit
    :param fidelity_threshold: Fidelity threshold above which to select candidates
    '''
    if verbose:
        if np.max(np.concatenate([cat_positive[:, int(i_SN)], cat_negative[:, int(i_SN)]])) > bin_edges[-1]:
            print("Warning, there are candidates above the max S/N.")

    # bins are the bin midpoints
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_N, _ = np.histogram(cat_negative[:, int(i_SN)], bins=bin_edges)
    hist_P, _ = np.histogram(cat_positive[:, int(i_SN)], bins=bin_edges)

    ### trim until the first bin that has measurements
    ind_first_N_clump = int(np.where(hist_N > 0)[0][0])
    hist_N = hist_N[ind_first_N_clump:]
    hist_P = hist_P[ind_first_N_clump:]
    bins = bins[ind_first_N_clump:]

    ### trim further to the min_SN_fit, to avoid the flattening of counts at low S/N
    ## the exact bins to be fitted could be further improved (see JGL+19)
    bin_minSN_i = np.where(bins > min_SN_fit)[0][0]
    hist_N_erf = lambda sn, c, sigma: c * (np.exp(- sn ** 2 / sigma ** 2))
    popt, _ = curve_fit(hist_N_erf,
                        xdata=bins[bin_minSN_i:],
                        ydata=hist_N[bin_minSN_i:],
                        sigma=np.sqrt(hist_N[bin_minSN_i:] + 1),
                        method='lm', maxfev=1000, p0=[1000, 1])

    hist_N_fitted = hist_N_erf(bins[bin_minSN_i:], popt[0], popt[1])
    # this is a mathematical trick (hist_P+1), but necessary for proper fit
    fidelity = 1 - np.clip(np.nan_to_num(hist_N_fitted / (hist_P[bin_minSN_i:] + 1), nan=0), 0, 1)
    # err_fidelity = np.clip(np.nan_to_num(np.sqrt(hist_N+1)/ (hist_P), nan=0), 0, 1)
    popt, pcorr = curve_fit(fidelity_function, xdata=bins[bin_minSN_i:], ydata=fidelity, method='lm', maxfev=1000, p0=[2, 5])

    sn_thres = erfinv((fidelity_threshold - 0.5) / 0.5) * popt[0] + popt[1]

    return bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, bins[bin_minSN_i:], hist_N_fitted


def fidelity_plot(cat_negative, cat_positive, bin_edges=np.arange(0, 20, 0.1),
                  i_SN=5, fidelity_threshold=0.6, plot_name='', title_plot=None,
                  min_SN_fit=4.0, verbose=True):

    bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, bins_fitted, hist_N_fitted \
        = fidelity_selection(cat_negative, cat_positive,
                             bin_edges,
                             i_SN, fidelity_threshold, min_SN_fit,
                             verbose=verbose)

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(211)
    # plot fidelity
    ax1.plot(bins_fitted, fidelity, drawstyle='steps-mid', c='C7')
    ax1.fill_between(bins_fitted, fidelity, step="mid", alpha=0.4, color='C7')
    # plot interpolated fidelity
    xr = np.linspace(bins[0], bins[-1], 200)
    ax1.plot(xr, fidelity_function(xr, popt[0], popt[1]), color='firebrick')
    # plot S/N thresh
    ax1.set_xlim(bins[0], bins[-1])
    ax1.set_ylim(-0.05, 1.05)
    ymin, ymax = ax1.get_ylim()
    plt.vlines(x=sn_thres, ymin=ymin, ymax=ymax, linestyles='--',
               label=f'F(S/N={sn_thres:.2f})={fidelity_threshold}', color='k')
    plt.xticks([])
    plt.ylabel('Fidelity')
    plt.legend()
    if title_plot is not None:
        plt.title(title_plot)

    ax2 = fig.add_subplot(212)
    # plot positive & negative hist
    ax2.plot(bins, hist_P, drawstyle='steps-mid')
    ax2.plot(bins, hist_N, drawstyle='steps-mid')
    ax2.plot(bins_fitted, hist_N_fitted, label='Negative Fit')
    ax2.set_yscale('log')
    ax2.fill_between(bins, hist_P, step="mid", alpha=0.4, label='Positive clumps')
    ax2.fill_between(bins, hist_N, step="mid", alpha=0.4, label='Negative clumps')
    ax2.legend(fontsize=10)
    # always show 1
    ax2.set_ylim(0.1, None)
    ymin, ymax = ax2.get_ylim()
    plt.vlines(x=sn_thres, ymin=ymin, ymax=ymax, linestyles='--', color='k')
    plt.ylabel(r'$\log N$')
    plt.xlabel('S/N')
    plt.legend()

    plt.subplots_adjust(wspace=None, hspace=None)

    if plot_name != '':
        plt.savefig(plot_name + '.pdf', bbox_inches="tight", dpi=300)

    return bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, fig, [ax1, ax2]


def fidelity_analysis(catN_name, catP_name,
                      bins=np.arange(0, 15, 0.2),
                      min_SN_fit=4.0,
                      fidelity_threshold=0.5,
                      kernels=np.arange(3, 20, 2),
                      verbose=True):
    """Perform and plot the fidelity analysis on a positive and negative catalog

    :param catN_name: name of the cropped catalog of negative lines
    :param catN_name: name of the cropped catalog of positive lines
    :param bins: SN bins for the fidelity analysis
    :param min_SN_fit: minimum SN for fidelity fit
    :param fidelity_threshold: Fidelity threshold above which to select candidates
    :param kernels: list of odd kernel widths to use

    :return: full list of positive candidates with fidelity
    :return: full list of negative candidates with fidelity
    :return: list of positive candidates with fidelity above `fidelity_threshold`
    :return: list of negative candidates with fidelity above `fidelity_threshold`
    """
    if verbose:
        print("Analysing fidelity.")
    # read in the input catalogs - these should already be cropped
    catN = Table.read(catN_name, format='ascii')
    catP = Table.read(catP_name, format='ascii')

    # first analyse fidelity for all kernels
    bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, _, _\
        = fidelity_plot(np.array([catN['SNR']]).T,
                        np.array([catP['SNR']]).T,
                        i_SN=0,
                        bin_edges=bins,
                        min_SN_fit=min_SN_fit,
                        fidelity_threshold=fidelity_threshold,
                        plot_name='hist_fid_all',
                        title_plot='all kernels', verbose=verbose)

    catP['F'] = fidelity_function(catP['SNR'], popt[0], popt[1])
    catN['F'] = -1 * np.ones_like(catN['SNR'])

    # analyse fidelity per kernel width
    # placeholder for the fidelity per kw
    catP['F_kw'] = -1 * np.ones_like(catP['SNR'])
    catN['F_kw'] = -1 * np.ones_like(catN['SNR'])

    for kernel in kernels:
        catP_kernel = catP[catP['BINNING'] == kernel]
        catN_kernel = catN[catN['BINNING'] == kernel]

        bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, _, _ \
            = fidelity_plot(np.array([catN_kernel['SNR']]).T,
                            np.array([catP_kernel['SNR']]).T,
                            i_SN=0,
                            bin_edges=bins,
                            min_SN_fit=min_SN_fit,
                            fidelity_threshold=fidelity_threshold,
                            plot_name=f'hist_fid_kw{kernel}',
                            title_plot=f'kernel = {kernel} ch', verbose=verbose)

        catP['F_kw'][catP['BINNING'] == kernel] = fidelity_function(catP['SNR'][catP['BINNING'] == kernel], popt[0], popt[1])

    # save output catalog sorted by SNR
    catP.sort("SNR", reverse=True)
    catN.sort("SNR", reverse=True)

    catP.add_column(np.arange(1, len(catP)+1), index=-1, name='ID')
    catN.add_column(np.arange(1, len(catN)+1), index=-1, name='ID')

    np.savetxt(catP_name.rstrip('.cat')+'_wfidelity_sorted.cat',
               catP,
               fmt=['%9.5f', '%9.5f', '%8.4f', '%5.1f', '%5.1f', '%6.2f',
                    '%9.6f', '%2.0f', '%2.0i', '%5.3f', '%5.3f', '%2.0i'],
               header='RA DEC FREQ_GHZ X Y SNR FLUX_MAX BINNING GROUP F F_kw ID')

    # return candidates above fidelity threshold
    candP = catP[np.where(catP['F_kw'] > fidelity_threshold)[0]]
    candN = catN[np.where(catN['F_kw'] > fidelity_threshold)[0]]

    return catP, catN, candP, candN
