Tools
*****

Interferopy comes with a toolbox of useful convenience functions that are either used in the Cube and MultiCube classes, or are meant to improve your analysis workflow.

These functions are divided in four broad categories: interferometry,  data analysis, mm/radio Astronomy helper functions for FindClumps.

First, let's import :py:mod:`interferopy.tools` as

.. code-block:: python

    import interferopy.tools as iftools



Interferometry
==============

The rms of 2D image can be computed using a custom rms routine (:any:`calcrms`) that excludes sources and artifacts via several rounds of sigma-clipping.

.. code-block:: python

    iftools.calcrms(arr=image_data_array, fitgauss=False, around_zero=True, clip_sigma=3, maxiter=20)

Another convenience function, :any:`beam_volume_sr` , computes the beam volume in steradian:

.. code-block:: python

    iftools.beam_volume_sr(bmaj=1, bmin=1) #bmin bmaj are the FWHM in arcsec

Finally, :any:`stack2d` helps with stacking radio images (in the image plane) at different RAs and Decs

.. code-block:: python

    stack2d(ras, decs, im, imhead, imrms=None, pathout=None, overwrite=False, naxis=100, interpol=True)

Additional specialized tasks and helper functions for data reduction in CASA are located in :any:`interferopy.casatools` and for VLA data :any:`interferopy.casatools_vla_pipe`. If you want to use these;

.. code-block:: python

    from interferopy.casatools import casareg
    from interferopy.casatools_vla_pipe import build_cont_dat, flagtemplate_add, lines_rest2obs, partition_cont_range


Data analysis
=============
A number of convenience functions are defined in to round/convert numbers:

.. code-block:: python

    iftools.sigfig(1.234, digits=3)  # rounds to 3 significant digits yielding 1.23
    width_ghz = iftools.kms2ghz(width_kms, freq_ghz)  # channel width in km/s to GHz at the reference frequency
    width_mhz = iftools.kms2mhz(width_kms,  freq_ghz) # velocity in km/s to MHz at the reference frequency
    width_kms = iftools.mhz2kms(width_mhz, freq_ghz) # width in  MHz to velocity in km/s at the reference frequency
    width_kms = iftools.ghz2kms(width_ghz, freq_ghz) #  width in  MHz to velocity in km/s at the reference frequency
    fwhm = iftools.sig2fwhm(sigma)  # convert Gaussian sigma to FWHM
    sigma = iftools.fwhm2sig(fwhm)  # and vice-versa
    kpc_per_arcsec = iftools.arcsec2kpc(z)  # 1 arcsec to kiloparsecs using concordence cosmology
    ra, dec = iftools.sex2deg(ra_hms, dec_dms)  # sexagesimal coordinates h:m:s and d:m:s (strings) to degrees

To help with the analysis of the data, weighted averaged and typical emission profiles are implemented:

.. code-block:: python

    avg, st_err, std = iftools.weighted_avg(values, weights) # returns the weighted average, standard error and deviation
    y = iftools.gauss(x, amp, freq0, sigma)  # Gaussian profile
    y = iftools.gausscont(x, cont, amp, freq0, sigma)  # Gaussian on top of a constant continuum profile

mm/radio Astronomy
==================

A number of function are implemented to compute surface brightness temperature, fit a modified blackbody to the dust SED and derive luminosities and star-formation rates

.. code-block:: python

    iftools.surf_temp(freq, rms, theta) #surface brightness temperature sensitivity in Kelvins from frequency [GHz], rms [Jy/beam] and beam FWHM [arcsec]
    iftools.dust_lum(nu_rest, Mdust, Tdust, beta) #intrinsic modified blackbody model (optically thin approx)
    iftools.dust_sobs(nu_obs, z, mass_dust, temp_dust, beta, cmb_contrast=True, cmb_heating=True) #observed dust SED following the prescriptions of DaCunha+2015
    iftools.dust_cont_integrate(dust_mass, dust_temp, dust_beta, print_to_console=True) # integrate a given MBB model and returns FIR/TIR luminosities and SFR rates using Kennicutt+1998,2012 conversions


Helper functions for FindClumps
===============================

:any:`line_stats_sextractor`, :any:`run_line_stats_sex`,  :any:`crop_doubles` functions are mostly run under the hood when using FindClumps on a :any:`Cube` and will be described in the appropriate section (:any:`findclumps`).

Once FindClumps has been run and candidate lines emitters have been found, the fidelity function can be built and used to define a SNR threshold above which emitters are considered real (:any:`tools.fidelity_selection`) and then plotted (:any:`tools.fidelity_plot`). By default, :any:`tools.fidelity_plot` calls :any:`tools.fidelity_selection` and thus supersedes it.

.. code-block:: python

    bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, hist_N_fitted = iftools.fidelity_selection(cat_negative, cat_positive, max_SN=20, i_SN=5, fidelity_threshold=0.6)
    bins, hist_N, hist_P, fidelity, popt, pcorr, sn_thres, fig, [ax1, ax2] = iftools.fidelity_plot(cat_negative, cat_positive, max_SN=20, i_SN=5, fidelity_threshold=0.6, plot_name='', title_plot=None)


Reference API
=============

.. automodapi:: interferopy.tools
   :no-inheritance-diagram:


.. automodapi:: interferopy.casatools
   :no-inheritance-diagram:

.. automodapi:: interferopy.casatools_vla_pipe
   :no-inheritance-diagram:
