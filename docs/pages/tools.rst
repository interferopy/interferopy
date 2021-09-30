Tools
=====

Statistics
----------

Additional helper functions are defined in [tools.py](src/tools.py), for example

.. code-block:: python

    iftools.sigfig(1.234, digits=3)  # rounds to 3 significant digits yielding 1.23
    iftools.calcrms(cub.im)  # calculate noise rms of the input array
    iftools.gausscont(x, cont, amp, freq0, sigma)  # Gaussian on top of a constant continuum profile

Various converter functions are defined here, for example

.. code-block:: python

    width_ghz = iftools.kms2ghz(width_kms, freq_ghz)  # channel width in km/s to GHz at the reference frequency
    fwhm = iftools.sig2fwhm(sigma)  # convert Gaussian sigma to FWHM
    kpc_per_arcsec = iftools.arcsec2kpc(z)  # 1 arcsec to kiloparsecs using concordence cosmology
    ra, dec = iftools.sex2deg(ra_hms, dec_dms)  # sexagesimal coordinates h:m:s and d:m:s (strings) to degrees
    
Additionally, there are several methods used for dust continuum calculations: *blackbody()* - Planck's law, *dust_lum()* - compute dust luminosity, *dust_sobs()* - compute observed flux density of dust. 
A method to compute the surface brightness temperature of radio observations: *surf_temp()*.
A method to stack different positions in a single 2D map: *stack2d()*, and others.

Interferometry
--------------

Several specialized tasks and helper functions for data reduction in CASA are located in [casatools.py](src/casatools.py) and [casatools_vla_pipe.py](src/casatools_vla_pipe.py).



Dust continuum
--------------


