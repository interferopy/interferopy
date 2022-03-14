Usage
*****

Interferopy is based on two classes handling your reduced sub(-mm)/radio datacubes produced by imaging interferometric data.
Two classes, *Cube* and *MultiCube*, are defined in :any:`Cube` and :any:`MultiCube`.  Note that datacubes can be 2D (images), or 3D (e.g. with a spectral dimension).

.. code-block:: python

    from interferopy.cube import Cube, MultiCube

Cube
====

Open a single data cube saved in fits format with

.. code-block:: python

    cub = Cube("image.fits")

**A 2D map is also considered a 3D cube, only with a single velocity channel.**
The Cube object gives quick access to useful properties and functions, for example

.. code-block:: python

    cub.freqs  # list of frequencies for each channel
    cub.vels  # list of velocities for each channel [optionally provide a reference frequency]
    cub.rms  # compute and return noise rms in each channel (usual units are Jy/beam)
    cub.beam  # beam sizes (resolution in arcsec), e.g., cub.beam["bmaj"]
    cub.pixsize  # pixel size in arcsec

Access to the image data is provided via

.. code-block:: python

    cub.im  # image, entire data cube
    cub.im[px, py, ch]  # a pixel value in a specific channel (one voxel)
    cub.im[px, py, :]  # single pixel spectrum along the cube
    cub.im[:, :, ch]  # 2D map of a single channel
    cub.head  # original fits image header

Converting between pixel and celestial coordinates (usually equatorial coordinate system, in the ICRS or J2000 epoch) is performed with

.. code-block:: python

    cub.wcs  # world coordinate system object
    px, py = cub.radec2pix(ra, dec)  # pixel to ra/dec coordinates (degrees)
    ra, dec = cub.pix2radec(px, py)
    ch = cub.freq2pix(freq)  # pixel (channel) to frequency coordinates
    freq = cub.pix2freq(ch)

Extract aperture flux density with the :any:`Cube.spectrum` method, for example

.. code-block:: python

    flux, err, n_pix, peak_sb = cub.spectrum(ra=ra, dec=dec, radius=1.5, calc_error=True) # r=1.5" aperture
    flux, err, _, _ = cub.spectrum(ra=ra, dec=dec, radius=0, calc_error=True) # single pixel spectrum extraction
    flux, err, _, _ = cub.spectrum(radius=1.5 calc_error=True) # center of the cube, r=1.5" aperture

Omitting the coordinates assumes the central pixel position, which is useful for a quick look at the data, especially in targeted observations, where the source is usually in the phase center. Setting the radius to zero yields a single pixel spectrum.

On top of the aperture-integrated flux and error,  :any:`Cube.spectrum` also returns the number of pixels in the aperture as well as the peak value in the aperture (which is especially useful to a peak SNR value in 2D maps).

When the :any:`Cube` loads a 2D image, :any:`Cube.spectrum` returns simply a single flux / error value.

Additional convenience wrapper functions exist (derived from the :any:`Cube.spectrum` method).

.. code-block:: python

    flux = cub.single_pixel_value()  # returns value(s) at the central pixel
    flux = cub.aperture_value(ra=ra, dec=dec, radius=0.5)  # integral within a circular aperture of r=0.5"
    flux, err = cub.aperture_value(ra=ra, dec=dec, radius=0.5, calc_error=True)

To find the best aperture size that encompasses the entire source, one can search for a saturation point in the curve of growth (cumulative flux density as a function of aperture radius). Obtain it with the` :any:`Cube.growing_aperture()` method.
This operates on a single channel only (must set the *freq* or the *channel* parameter). The same function implements the ability to compute azimuthally averaged radial profile.

.. code-block:: python

    r, flux, err, _ = cub.growing_aperture(ra=ra, dec=dec, freq=freq, maxradius=5, calc_error=True)
    r, profile, err, _ = cub.growing_aperture(ra=ra, dec=dec, freq=freq, calc_error=True, profile=True)

Again, convenience wrapper functions exist (derived from the :any:`Cube.growing_aperture` method).

.. code-block:: python

    r, flux = cub.aperture_r()  # use the central pixel, the first channel, and the maxradius of 1" by default
    r, profile = cub.profile_r()

MultiCube
=========

During the imaging process (e.g., using CASA task *tclean*), several cubes are produced, which all pertain to the same dataset and the same observed source.
The :any:`MultiCube` is a container, a dictionary-like class that can hold multiple cubes simultaneously. This class also defines functions that operate on multiple cubes, such as the primary beam correction or the residual scaled aperture integration (see Appendix A of `Novak et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...881...63N/abstract>`_ , and references therein).

Loading the :any:`MultiCube` object is performed  with

.. code-block:: python

     mcub = MultiCube("image.fits")

If a specific naming convention is used, it will load automatically other available cubes in the same directory, such as the *image.dirty.fits*, *image.residual.fits*, *image.pb.fits*, and so on. This behavior can be overriden, and the cubes can be loaded manually

.. code-block:: python

     mcub = MultiCube("image.fits", autoload_multi=False)  # open the first map
     # mcub = MultiCube()  # alternatively, intialize an empty container
     # mcub.load_cube("/somewhere/cube.fits", "image")
     mcub.load_cube("/elsewhere/cube.dirty.fits", "dirty")
     mcub.load_cube("/elsewhere/cube.residual.fits", "residual")

Specific cubes are accessed via their keys:

.. code-block:: python

    mcub.loaded_cubes  # list of loaded cubes
    cub = mcub["image"]  # return a Cube object
    cub = mcub.cubes["image"]  # same as above

Analogous to :any:`Cube.spectrum` and :any:`Cube.growing_aperture` methods available for a :any:`Cube` object,
the *MultiCube* object has :any:`MultiCube.spectrum_corrected` and :any:`MultiCube.growing_aperture_corrected`. These methods perform aperture integration that takes into account the ill-defined hybrid units of the cleaned maps. They require loaded *image*, *residual*, and *dirty* cubes (best to have the *pb* cube as well).

.. code-block:: python

    flux, err, tab = mcub.spectrum_corrected(ra=ra, dec=dec, radius=1.5, calc_error=True)
    r, flux, err, tab = mcub.growing_aperture_corrected(ra=ra, dec=dec, maxradius=5, calc_error=True)

These methods perform both the residual scaling correction, and the primary beam correction (can be turned off with *apply_pb_corr=False*). The *tab* will contain a *Table* object with additional technical information, such as the aperture integrated values from individual cubes, the clean-to-dirty beam ratio, number of pixels or beams in the aperture, and so on.

Reference API
=============

.. automodapi:: interferopy.cube
   :no-inheritance-diagram:
   :skip: tqdm
   :skip: Table
