Finding line emitters with Findclumps
=====================================

Interferopy includes an implementation of the Findclumps algorithm used by Walter+2016 to find line emitters in ASPECS (https://ui.adsabs.harvard.edu/abs/2016ApJ...833...67W/abstract).
At its core, Findclumps simply convolves the cube with boxcar kernels of various sizes, and run sextractor on the created image to find "clumps". It does so on the original and inverted cubes, enabling the user to estimate which detections are real or not. It groups "clumps" by frequency and spatial distance, at the discretion of the user. 

To run FindClcumps, you will need to have sextractor installed (which can be done via an astroconda environment : https://astroconda.readthedocs.io/en/latest/package_manifest.html), and have a local "default.sex" file in the folder where you run the interferopy-findclumps script. You can either copy your generic default.sex that comes as  part of sextractor of modify it to optimise the search for "clumps" in the cube.

Example usage:

.. code-block:: python
    
    from interferopy.cube import Cube
    cube = Cube('absolute_path/filename')
    cube.findclumps_full(output_file='findclumps_', kernels=np.arange(3, 20, 2), rms_region=1. / 4.,
                        sextractor_param_file='default.sex', clean_tmp=True, min_SNR=5,
                        delta_offset_arcsec=2, delta_freq=0.1, ncores=1,
                        run_positive=True, run_negative=True,
                        verbose=False)

    cat_Neg = np.loadtxt("findlucmps_clumpsN_minSNR_5.cat")
    cat_Pos = np.loadtxt("findlucmps_clumpsN_minSNR_5.cat")

    sn_threshold = iftools.fidelity_selection(cat_negative=cat_Neg, cat_positive=cat_Pos, i_SN=5,
                        fidelity_threshold=0.6,plot_name='./clumps/fidelity_selection_FINDCLUMPS.pdf')

    candidates = cat_Pos[np.where(cat_Pos[:,5]>sn_threshold)[0]] # selecting candidates for future use
    np.savetxt('saved_high_fidelity_candidates',candidates)

See :any:`interferopy.cube.findclumps_full`, :any:`interferopy.cube.findclumps_1kernel`, :any:`interferopy.tools.fidelity_selection` for more details.

The resulting fidelity selection plot should look like this:

.. image:: ../../examples/thumbnails/fidelity_example.png

