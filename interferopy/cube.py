import os
from tqdm import tqdm
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import scipy.constants as const
import interferopy
import interferopy.tools as tools


class Cube:
    """
    Cube object represents interferometric data such as a map (2D) or a cube (3D).
    Allows performing tasks such as aperture integration or spectrum extraction.
    It is instantiated by reading in a fits file.

    :Example:

    .. code-block:: python

        filename="cube.image.fits"
        ra, dec, freq = (205.533741, 9.477317341, 222.54) # coord for the emission line
        c=Cube(filename)
        # spectrum from the circular aperture, with associated error
        flux, err, _ = c.spectrum(ra=ra, dec=dec, radius=1.5, calc_error=True)
        # curve of growth up to the maximum radius, in steps of one pixel, in a chosen frequency channel
        radius, flux, err, _ = c.growing_aperture(ra=ra, dec=dec, freq=freq, maxradius=5, calc_error=True)
    """

    def __init__(self, filename: str):
        """
        :param filename: Path string to the fits image.
        """
        if os.path.exists(filename):
            self.filename = filename
            self.hdu = None
            """Hdu of the loaded fits file."""

            self.head = None
            """Image header of the loaded fits file."""

            self.im: np.ndarray = None
            """Data cube indexed as im[ra, dec, freq]. This is transposed version of what the fits package returns"""

            self.wcs: wcs.WCS = None
            """Datacube world coordinate system."""

            self.beam = None
            """Table of beams per channel, keywords: bmaj, bmin, bpa."""

            self.beamvol: np.ndarray = None
            """Beam volume in pixels (number of pixels in a beam)."""

            self.pixsize: float = None
            """Size of the pixel in arcsec."""

            self.nch: int = None
            """Number of channels in a cube."""

            self.freqs: np.ndarray = None
            """Array of frequencies corresponding to cube channels."""

            self.reffreq: float = None
            """Reference frequency in GHz (from image header)."""

            self.deltafreq: float = None
            """Channel width in GHz."""

            self.__rms: np.ndarray = None
            """Array of rms values per channel. Computed on demand."""
            self.__load_fitsfile()
        else:
            raise FileNotFoundError(filename)

    def __load_fitsfile(self):
        """
        Read the fits file and extract common useful data into Cube class attributes.
        """
        self.log("Open " + self.filename)
        self.hdu = fits.open(self.filename)
        self.head = self.hdu[0].header
        self.pixsize = self.head["cdelt2"] * 3600  # arcsec, assumes square pixels

        # quick fix for a faulty header card that contained \n char - remove the tag altogether
        if "ORIGIN" in self.head.keys():
            self.head.remove('ORIGIN')

        # transpose the cube data so that intuitive im[ra,dec,freq,stokes] indexing can be used
        # number of cube dimensions
        naxis = len(self.hdu[0].data.shape)
        if naxis == 4 and self.hdu[0].data.shape[0] == 1:
            # drop the fourth (Stokes) axis if only a single plane is present
            image = self.hdu[0].data.T[:, :, :, 0]
            naxis = 3
        elif naxis == 4:
            image = self.hdu[0].data.T
            self.log("Warning: did test 4D full polarization cubes with multiple Stokes axes.")
        elif naxis == 3:
            image = self.hdu[0].data.T
        elif naxis == 2:
            # add the third axis, if missing, because several methods require it
            image = self.hdu[0].data[np.newaxis, :, :].T
            # add also in saved header in Cube
            self.head['NAXIS3'] = 1
            self.head['NAXIS'] = 3
            self.head['CTYPE3'] = 'FREQ'
            self.head['CRVAL3'] = self.head['RESTFREQ']
            self.head['CDELT3'] = 1
            self.head['CRPIX3'] = 0
            naxis = 3
            self.log("Warning: 2D maps may have incorrect pixsize/beamvol. Please replace!")
        else:
            raise RuntimeError("Invalid number of cube dimensions.")
        self.im = image
        self.naxis = naxis

        if not np.isnan(self.im[(0,) * self.naxis]):
            self.log("Warning: The first voxel of the cube is not np.nan. "
                     "Note regions with no data are assumed to have value np.nan. "
                     "If needed, use self.im_mask_values to mask appropriately.")

        # save the world coord system
        self.wcs = wcs.WCS(self.head, naxis=self.naxis)

        # populate frequency details (freq array, channel size, number of channels)
        # convert from velocity header if necessary, scale to GHz
        nch = 1
        if naxis >= 3:
            nch = self.im.shape[2]  # number of (freq) channels in the cube
            # if frequencies are already 3rd axis
            if str(self.head["CTYPE3"]).strip().lower() == "freq":
                _, _, freqs = self.wcs.all_pix2world(int(self.im.shape[0] / 2), int(self.im.shape[1] / 2), range(nch),
                                                     0)
                freqs *= 1e-9  # in GHz
                self.deltafreq = self.head["CDELT3"] * 1e-9
                self.reffreq = self.head["CRVAL3"] * 1e-9
            # if frequencies are given in radio velocity, convert to freqs
            elif str(self.head["CTYPE3"]).strip().lower() == "vrad":
                _, _, vels = self.wcs.all_pix2world(int(self.im.shape[0] / 2), int(self.im.shape[1] / 2), range(nch), 0)
                if "RESTFREQ" in self.head.keys():
                    reffreq = self.head["RESTFREQ"] * 1e-9
                elif "RESTFRQ" in self.head.keys():
                    reffreq = self.head["RESTFRQ"] * 1e-9
                else:
                    # unknown
                    reffreq = 0
                freqs = reffreq * (1 - vels / const.c)  # in GHz
                self.reffreq = reffreq

                if "CUNIT3" in self.head.keys() and self.head["CUNIT3"].strip().lower() == "km/s":
                    vel_scale = 1000
                else:
                    vel_scale = 1
                self.deltafreq = reffreq * (self.head["CDELT3"] * vel_scale / const.c)
            elif str(self.head["CTYPE3"]).strip() == "WAVE":
                _, _, wave_lambda = self.wcs.all_pix2world(int(self.im.shape[0] / 2), int(self.im.shape[1] / 2),
                                                           range(nch), 0)
                if str(self.head["CUNIT3"]).strip().lower() == "m":
                    freqs = 3e5 / wave_lambda * 1e-9  # in GHz
                    self.deltafreq = 3e5 / self.head["CDELT3"] * 1e-9
                    self.reffreq = 3e5 / self.head["CRVAL3"] * 1e-9
                else:
                    freqs = None
                    self.log("Warning: unknown 3rd axis units.")
            # if frequencies are given in radio velocity, convert to freqs
            else:
                print(str(self.head["CTYPE3"]))
                freqs = None
                self.log("Warning: unknown 3rd axis format.")
            self.freqs = freqs
        self.nch = nch

        # populate beam data
        # single beam from the main header
        if "BMAJ" in self.head.keys() and "BMIN" in self.head.keys() and "BPA" in self.head.keys():
            beam = {"bmaj": self.head["BMAJ"] * 3600, "bmin": self.head["BMIN"] * 3600, "bpa": self.head["BPA"]}
        else:
            beam = {"bmaj": 0, "bmin": 0, "bpa": 0}

        # if there is a beam per channel table present, take it
        if nch > 1 and len(self.hdu) > 1:
            beam_table = self.hdu[1].data  # this is inserted by CASA, there is unit metadata, but should be arcsec
        # otherwise clone the single beam across all channels in a new table
        else:
            if len(self.im.shape) >= 3:
                nch = self.im.shape[2]
            else:
                nch = 1

            beam_table = Table()
            beam_table.add_columns([Table.Column(name='bmaj', data=np.ones(nch) * beam["bmaj"])])
            beam_table.add_columns([Table.Column(name='bmin', data=np.ones(nch) * beam["bmin"])])
            beam_table.add_columns([Table.Column(name='bpa', data=np.ones(nch) * beam["bpa"])])
            beam_table.add_columns([Table.Column(name='chan', data=range(nch))])
            beam_table.add_columns([Table.Column(name='pol', data=np.zeros(nch))])
        self.beam = beam_table

        # model image is Jy/pixel, and since beam volume divides integrated fluxes, set it to 1 in this case
        if "BUNIT" in self.head.keys() and self.head["BUNIT"].strip().lower().endswith("/pixel"):
            self.beamvol = np.array([1] * nch)
        else:
            # "BUNIT" in self.head.keys() and self.head["BUNIT"].strip().lower().endswith("/beam"):
            # sometimes BUNIT is not given, although beam is in the header (e.g. the PSF map)
            self.beamvol = np.pi / (4 * np.log(2)) * self.beam["bmaj"] * self.beam["bmin"] / self.pixsize ** 2
            if type(self.beamvol) is Table.Column:
                self.beamvol = self.beamvol.data

    def write_fitsfile(self, filename: str, overwrite=False):
        """
        Write the current head and im to a new fitsfile (useful if modified).
        This is still experimental and may fail on certain headers.

        :param filename: Path string to the output file.
        :param overwrite: False by default.
        """
        # transpose the cube back
        hdu = fits.PrimaryHDU(data=self.im.T, header=self.head)
        hdu.writeto(filename, overwrite=overwrite)


    def log(self, text: str):
        """
        Basic logger function to allow better functionality in the future development.
        All class functions print info through this wrapper.
        Could be extended to provide different levels of info, timestamps, or logging to a file.
        """
        print(text)

    def im_mask_values(self, value_to_mask=None, mask_value=np.nan):
        """Mask specific `values_to_mask` in the image to `mask_value`.

        For example, if regions with no data have values equal to zero
        or another fill value they can be set to np.nan.  This is
        required for `get_rms()` to work correctly.

        :param value_to_mask: value to set to `mask_value`, if `None` the value of the first pixel is used `
        :param mask_value: the value of pixels to be masked (default: np.nan)
        """
        if value_to_mask is None:
            # if not given, it will assumes the first value is the value
            # to be masked, but only if it is the same as the last value
            out = self.im[(0,)*self.naxis]
            out2 = self.im[(-1,)*self.naxis]
            assert out == out2, "Cannot safely determine value_to_mask and it was not provided."
        else:
            out = value_to_mask

        if out != mask_value:
            self.im[self.im == out] = mask_value
        else:
            print ("Warning: You are masking to the same value.")

    def im_mask_channels(self, channels_to_mask, mask_value=np.nan):
        """Mask specific `channels_to_mask` to `mask_value`.

        :param channels_to_mask: list of channels to be masked (python indexing)
        :param mask_value: the value of pixels to be masked (default: np.nan)
        """

        for chan in channels_to_mask:
            self.im[:,:,chan] = mask_value

    def get_rms(self):
        """
        Calculate rms for each channel of the cube. Can take some time to compute on large cubes.

        :return: single rms value if 2D, array odf rms values for each channel if 3D cube
        """
        # TODO: maybe add the option to calculate a single channel only?
        if self.__rms is None:
            # calc rms per channel
            if self.nch > 1:
                self.__rms = np.zeros(self.nch)
                self.log("Computing rms in each channel.")
                for i in range(self.nch):
                    self.__rms[i] = tools.calcrms(self.im[:, :, i])
            else:
                self.log("Computing rms.")
                self.__rms = np.array([tools.calcrms(self.im)])
        return self.__rms

    def set_rms(self, value):
        self.__rms = value

    rms = property(get_rms, set_rms)
    """Rms (root mean square) per channel."""

    def deltavel(self, reffreq: float = None):
        """
        Compute channel width in velocity units (km/s).

        :param reffreq: Computed around specific velocity. If empty, use referent one from the header.
        :return: Channel width in km/s. Sign reflects how channels are ordered.
        """

        if reffreq is None:
            reffreq = self.reffreq

        return self.deltafreq / reffreq * const.c / 1000  # in km/s

    def vels(self, reffreq: float):
        """
        Compute velocities of all cube channels for a given reference frequency.

        :param reffreq: Reference frequency in GHz. If empty, use referent one from the header.
        :return: Velocities in km/s.
        """

        if reffreq is None:
            reffreq = self.reffreq

        return const.c / 1000 * (1 - self.freqs / reffreq)

    def radec2pix(self, ra=None, dec=None, integer=True):
        """
        Convert ra and dec coordinates into pixels to be used as im[px, py].
        If no coords are given, the center of the map is assumed.

        :param ra: Right ascention in degrees. Or list.
        :param dec: Declination in degrees. Or list.
        :param integer: Round the coordinate to an integer value (to be used as indices).
        :return: Coords x and y in pixels (0 based index).
        """

        # use the central pixel if no coords given
        if ra is None or dec is None:
            px = self.im.shape[0] / 2
            py = self.im.shape[1] / 2
        # otherwise convert radec to pixel coord
        else:
            if len(self.wcs.axis_type_names) < 3:
                px, py = self.wcs.all_world2pix(ra, dec)
            else:
                px, py, _ = self.wcs.all_world2pix(ra, dec, self.freqs[0], 0)

        # need integer indices
        if integer:
            px = np.asarray(np.round(px), dtype=int)
            py = np.asarray(np.round(py), dtype=int)

        return px, py

    def pix2radec(self, px=None, py=None):
        """
        Convert pixels coordinates into ra and dec.
        If no coords are given, the center of the map is assumed.

        :param px: X-axis index. Or list.
        :param py: Y-axis index. Or list.
        :return: Coords ra, dec in degrees
        """
        if px is None or py is None:
            px = self.im.shape[0] / 2
            py = self.im.shape[1] / 2

        if len(self.wcs.axis_type_names) < 3:
            ra, dec = self.wcs.all_pix2world(px, py)
        else:
            ra, dec, _ = self.wcs.all_pix2world(px, py, self.freqs[0], 0)

        return ra, dec

    def freq2pix(self, freq: float = None):
        """
        Get the channel number of requested frequency.

        :param freq: Frequency in GHz.
        :return: Channel index.
        """

        if len(self.wcs.axis_type_names) < 3:
            raise ValueError("No frequency axis is present.")
            return None

        if freq is None:
            pz = self.im.shape[2] / 2
            pz = int(round(pz))
            return pz

        if freq < np.min(self.freqs) - 0.5 * np.abs(self.deltafreq) \
                or freq > np.max(self.freqs) + 0.5 * np.abs(self.deltafreq):
            raise ValueError("Requested frequency is outside of the available range.")
            return None

        # This is ok, unless the 3rd axis is in velocities
        # ra0 = self.head["CRVAL1"]
        # dec0 = self.head["CRVAL2"]
        # _, _, pz = self.wc.all_world2pix(ra0, dec0, freq * 1e9, 0)
        # pz = int(np.round(pz))  # need integer index

        # freqs is populated when the cube is loaded, regardless whether the header is in vels or freqs
        # find the index with the minimum offset
        pz = np.argmin(np.abs(self.freqs - freq))

        return pz

    def pix2freq(self, pz: int = None):
        """
        Get the frequency of a given channel.
        If no channel is given, the center channel is assumed.

        :param pz: Channel index.
        :return: Frequency in GHz.
        """

        if len(self.wcs.axis_type_names) < 3:
            raise ValueError("No frequency axis is present.")
            return None

        if pz is None:
            pz = self.im.shape[2] / 2
            pz = int(round(pz))

        if pz < 0 or pz >= len(self.freqs):
            raise ValueError("Requested channel is outside of the available range.")
            return None

        return self.freqs[pz]

    def im2d(self, ch: int = None, freq: float = None, function=np.sum):
        """Get a 2D map. Convenience function to avoid indexing notation.

        Provided with a single `ch` (or `freq`, but not both) it will
        return that channel. Alternatively, `ch` or `freq` can also be
        a list with the `[start, stop]` of a slice, to which
        `function` is then applied to make a collapsed image (default:
        `np.sum`).

        :param ch: channel index, alternatively: a list with a slice
            `[start, stop]`` to which `function` is applied
        :param freq: channel freq, alternatively: a list with a slice
            `[start, stop]`` to which `function` is applied
        :param function: function to apply when `ch` or `freq` is a slice (default: `np.sum`),
            should be a (numpy) function that takes the argument `axis=-1` and aggregates over it
        :return: 2D numpy array.

        """
        if ch is not None and freq is not None:
            raise ValueError("Provide either channel or frequency (not both).")
            return None

        try:
            # multiple frequencies? convert to channels
            ch_tmp = []
            for i, f in enumerate(freq):
                ch_tmp.append(self.freq2pix(f))
            ch = ch_tmp
        except TypeError:
            # still try multiple channels
            pass

        try:
            # multiple channels?
            for c in ch:
                if abs(c) >= self.nch:  # allow negative channel indexing
                    raise ValueError("Requested channel is outside of the available range.")
                    return None

            return function(self.im[:, :, ch[0]:ch[1]], axis=-1)

        except TypeError:
            # single channel or frequency
            if freq is not None:
                ch = self.freq2pix(freq)
            elif abs(ch) >= self.nch:  # allow negative channel indexing
                raise ValueError("Requested channel is outside of the available range.")
                return None

        return self.im[:, :, ch]

    def distance_grid(self, px: float, py: float) -> np.ndarray:
        """
        Grid of distances from the chosen pixel (can be subpixel accuracy).
        Uses small angle approximation (simple Pythagorean distances).
        Distances are measured in numbers of pixels.

        :param px: Index of x coord.
        :param py: Index of y coord.
        :return: 2D grid of distances in pixels, same shape as the cube slice
        """

        xxx = np.arange(self.im.shape[0])
        yyy = np.arange(self.im.shape[1])
        distances = np.sqrt((yyy[np.newaxis, :] - py) ** 2 + (xxx[:, np.newaxis] - px) ** 2)
        return distances

    def spectrum(self, ra: float = None, dec: float = None, radius=0.0,
                 px: int = None, py: int = None, channel: int = None, freq: float = None, calc_error=False) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Extract the spectrum (for 3D cube) or a single flux density value (for 2D map) at a given coord (ra, dec)
        integrated within a circular aperture of a given radius.
        Coordinates can be given in degrees (ra, dec) or pixels (px, py).
        If no coordinates are given, the center of the map is assumed.
        Single plane can be chosen instead of full spectrum by defining freq or channel.
        If no radius is given, a single pixel value is extracted (usual units Jy/beam), otherwise aperture
        integrated spectrum is extracted (usual units of Jy).

        Note: use the freqs field (or velocities method) to get the x-axis values.

        :param ra: Right ascention in degrees.
        :param dec: Declination in degrees.
        :param radius: Circular aperture radius in arcsec.
        :param px: Right ascention pixel coord.
        :param py: Declination pixel coord.
        :param channel: Force extracton in a single channel of provided index (instead of the full cube).
        :param freq: Frequency in GHz (alternative to channel).
        :param calc_error: Set to False to skip error calculations, if the rms computation is slow or not necessary.
        :return: flux, err, npix: flux (spectrum), error estimate, and number of pixels in the aperture
        """

        if px is None or py is None:
            px, py = self.radec2pix(ra, dec)
        if freq is not None:
            channel = self.freq2pix(freq)

        # take single pixel value if no aperture radius given
        if radius <= 0:
            self.log("Extracting single pixel spectrum.")
            flux = self.im[px, py, :]
            if calc_error:
                err = np.array(self.rms[:])  # single pixel error is just rms
            else:
                err = np.full_like(flux, np.nan)
            npix = np.ones(len(flux))
            # use just a single channel
            if channel is not None:
                flux = np.array([flux[channel]])
                npix = np.array([npix[channel]])
                err = np.array([err[channel]])
            peak_sb = flux
        else:
            self.log("Extracting aperture spectrum.")
            # grid of distances from the source in arcsec, need for the aperture mask
            distances = self.distance_grid(px, py) * self.pixsize

            # select pixels within the aperture
            w = distances <= radius

            if channel is not None:
                npix = np.array([np.sum(np.isfinite(self.im[:, :, channel][w]))])
                flux = np.array([np.nansum(self.im[:, :, channel][w]) / self.beamvol[channel]])
                peak_sb = np.nanmax(self.im[:, :, channel][w])
                if calc_error:
                    err = np.array(self.rms[channel] * np.sqrt(npix / self.beamvol[channel]))
                else:
                    err = np.array([np.nan])
            else:
                flux = np.zeros(self.nch)
                peak_sb = np.zeros(self.nch)
                npix = np.zeros(self.nch)
                for i in range(self.nch):
                    flux[i] = np.nansum(self.im[:, :, i][w]) / self.beamvol[i]
                    peak_sb[i] = np.nanmax(self.im[:, :, i][w])
                    npix[i] = np.sum(np.isfinite(self.im[:, :, i][w]))
                if calc_error:
                    err = np.array(self.rms * np.sqrt(npix / self.beamvol))
                else:
                    err = np.full_like(flux, np.nan)

        return flux, err, npix, peak_sb

    def single_pixel_value(self, ra: float = None, dec: float = None, freq: float = None,
                           channel: int = None, calc_error: bool = False):
        """
        Get a single pixel value at the given coord. Optionally, return associated error.
        Wrapper function for the "spectrum" method. If None, assumes central pixel.

        :param ra: Right ascention in degrees.
        :param dec: Declination in degrees.
        :param freq: Frequency in GHz. If None, computes whole spectrum.
        :param channel: Channel index (alternative to freq).
        :param calc_error: If true, returns also an error estimate (rms).
        :return: value or (value, error)
        """

        spec, err, _, _ = self.spectrum(ra=ra, dec=dec, radius=0, freq=freq, channel=channel, calc_error=calc_error)
        if calc_error:
            return spec, err
        else:
            return spec

    def aperture_value(self, ra: float = None, dec: float = None, freq: float = None, channel: int = None,
                       radius: float = 1.0, calc_error=False):
        """
        Get an aperture integrated value at the given coord. Optionally, return associated error.
        Wrapper function for the "spectrum" method.

        :param ra: Right ascention in degrees. If None, assumes central pixel.
        :param dec: Declination in degrees.
        :param freq: Frequency in GHz. If None, computes whole spectrum.
        :param channel: Channel index (alternative to freq).
        :param radius: Aperture radius in arcsec.
        :param calc_error: If true, returns also an error estimate (rms x sqrt(num of beams in aperture)).
        :return: value or (value, error)
        """

        flux, err, _, _ = self.spectrum(ra=ra, dec=dec, radius=radius, freq=freq, channel=channel,
                                        calc_error=calc_error)
        if calc_error:
            return flux, err
        else:
            return flux

    def growing_aperture(self, ra: float = None, dec: float = None, freq: float = None,
                         maxradius=1.0, binspacing: float = None, bins: list = None,
                         px: int = None, py: int = None, channel: int = 0,
                         profile=False, calc_error=False) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute curve of growth at the given coordinate position in a circular aperture, growing up to the max radius.
        Coordinates can be given in degrees (ra, dec) or pixels (px, py).
        If no coordinates are given, the center of the map is assumed.
        If no channel is provided, the first one is assumed.

        :param ra: Right ascention in degrees.
        :param dec: Declination in degrees.
        :param freq: Frequency in GHz, takes precedence over channel param.
        :param maxradius: Max radius for aperture integration in arcsec.
        :param binspacing: Resolution of the growth flux curve in arcsec, default is one pixel size.
        :param bins: Custom bins for curve growth (1D np array).
        :param px: Right ascention pixel coord (alternative to ra).
        :param py: Declination pixel coord (alternative to dec).
        :param channel: Index of the cube channel to take (alternative to freq).
        :param profile: If True, compute azimuthally averaged profile, if False, compute cumulative aperture values
        :param calc_error: Set to False to skip error calculations, if the rms computation is slow or not necessary.
        :return: radius, flux, err, npix - all 1D numpy arrays: aperture radius, cumulative flux within it,
            associated Poissionain error (based on number of beams inside the aprture and the map rms), number of pixels
        """
        self.log("Running growth_curve.")

        # get coordinates in pixels
        if px is None or py is None:
            px, py = self.radec2pix(ra, dec)
        # get grid of distances from the coordinate
        distances = self.distance_grid(px, py) * self.pixsize

        if freq is not None:
            channel = self.freq2pix(freq)

        if bins is None:
            if binspacing is None:
                # take one pixel size as default size
                binspacing = self.pixsize
            bins = np.arange(0, maxradius, binspacing)

        # histogram will fail if nans are present, select only valid pixels
        w = np.isfinite(self.im[:, :, channel])

        # histogram by default sums the number of pixels
        npix = np.array(np.histogram(distances[w], bins=bins)[0])
        if profile:
            pass
        else:
            npix = np.array(np.cumsum(npix))

        # pixel values are added as histogram weights to get the sum of pixel values
        flux = np.histogram(distances[w], bins=bins, weights=self.im[:, :, channel][w])[0]
        if profile:
            # mean value - azimuthally averaged
            flux = np.array(flux / npix)
        else:
            # cumulative flux inside an aperture is the sum of all pixel values divided by the beam volume
            flux = np.cumsum(flux) / self.beamvol[channel]

        if calc_error:
            if profile:
                # error on the mean
                nbeam = npix / self.beamvol[channel]
                nbeam[nbeam < 1] = 1  # if there are less pixels than the beam size, better not count less than 1
                err = np.array(self.rms[channel] / np.sqrt(nbeam))
            else:
                # error estimate assuming Poissonian statistics: rms x sqrt(number of independent beams inside aperture)
                nbeam = npix / self.beamvol[channel]
                nbeam[nbeam < 1] = 1
                err = np.array(self.rms[channel] * np.sqrt(nbeam))
        else:
            err = np.full_like(flux, np.nan)

        # centers of bins
        radius = np.array(0.5 * (bins[1:] + bins[:-1]))

        # old loop version, human readable, but super slow in execution for large apertures and lots of pixels
        # for i in range(len(bins)-1):
        # 	#w=(distances>=bins[i]) & (distances<bins[i+1]) #annulus
        # 	w=(distances>=0) & (distances<bins[i+1]) # aperture (cumulative)
        # 	npix[i]=np.sum(w)
        # 	flux[i]=np.sum(im[w])/beamvol
        # 	err[i]=rms*np.sqrt(npix[i]/beamvol) # rms times sqrt of beams used for integration

        return radius, flux, err, npix

    def aperture_r(self, ra: float = None, dec: float = None, freq: float = None,
                   px: int = None, py: int = None, channel: int = 0,
                   maxradius: float = 1.0, binspacing: float = None, bins: list = None, calc_error: bool = False):
        """
        Obtain integrated flux within a circular aperture as a function of radius.
        If freq is undefined, will return the spectrum of the 3D cube.
        Alias function. Check the "growing_aperture" method for details.

        Units: ra[deg], dec[deg], maxradius[arcsec], binspacing[arcsec], freq[GHz]

        :return: (radius, flux) or (radius, flux, err) if calc_error
        """

        radius, flux, err, npix = self.growing_aperture(ra=ra, dec=dec, freq=freq,
                                                        maxradius=maxradius, binspacing=binspacing, bins=bins,
                                                        px=px, py=py, channel=channel,
                                                        profile=False, calc_error=calc_error)
        if calc_error:
            return radius, flux, err
        else:
            return radius, flux

    def profile_r(self, ra: float = None, dec: float = None, freq: float = None,
                  px: int = None, py: int = None, channel: int = 0,
                  maxradius: float = 1.0, binspacing: float = None, bins: list = None,
                  calc_error: bool = False):
        """
        Obtain azimuthaly averaged profile as a function of radius.
        Alias function. Check the "growing_aperture" method for details.

        Units: ra[deg], dec[deg], maxradius[arcsec], binspacing[arcsec], freq[GHz]

        :return: (radius, profile) or (radius, profile err) if calc_error
        """
        radius, profile, err, npix = self.growing_aperture(ra=ra, dec=dec, freq=freq,
                                                           maxradius=maxradius, binspacing=binspacing, bins=bins,
                                                           px=px, py=py, channel=channel,
                                                           profile=True, calc_error=calc_error)
        if calc_error:
            return radius, profile, err
        else:
            return radius, profile

        return

    def findclumps_1kernel(self, output_file, rms_region=1./4., minwidth=3,
                           sextractor_param_file='',
                           clean_tmp=True, negative=False, verbose=False):
        """
        FINDCLUMP(s) algorithm (Decarli+2014,Walter+2016). Takes the cube image and outputs the 3d (x,y,wavelength) position
        of clumps of a minimum SN specified. Works by using a top-hat filter on a rebinned version of the datacube.

        :param output_file: relative/absolute path to the output catalogue
        :param rms_region: Region to compute the rms noise [2x2 array in image pixel coord].
            If ``None``, takes the central 25% pixels (square)
        :param minwidth: Number of channels to bin
        :param sextractor_param_file: if '', gets the default.sex file in interferopy
        :param clean_tmp: Whether to remove or not the temporary files created by Sextractor
        :return:
        """
        # keep tmpdir in the same folder as output file
        basepath = os.path.split(output_file)[0]
        if negative:
            tmpdir = os.path.join(basepath, 'tmp_findclumps_kw' + str(minwidth) + '_N/')
        else:
            tmpdir = os.path.join(basepath, 'tmp_findclumps_kw' + str(minwidth) + '_P/')
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        assert not (os.path.isfile(output_file + '_kw' + str(
            int(minwidth)) + '.cat')), 'Output file "' + output_file + '_kw' + str(
            int(minwidth)) + '" already exists - please delete or change name!'
        if minwidth % 2 == 1:
            chnbox = int((minwidth - 1) / 2.0)
        else:
            self.log('WARNING: Window must be odd number of channels, using minwidth +1')
            chnbox = int(minwidth / 2.0)

        if negative:
            # sextractor expects the counter-intuitive freq,dec,ra order (because the header has not been changed)
            cube = -self.im.T
        else:
            # sextractor expects the counter-intuitive freq,dec,ra order (because the header has not been changed)
            cube = self.im.T

        nax1 = cube.shape[0]
        nax2 = cube.shape[1]
        nax3 = cube.shape[2]
        if verbose:
            self.log("Running sextractor for kernel_width={}".format(minwidth))
            iterable = tqdm(range(chnbox, nax1 - chnbox - 1))
        else:
            iterable = range(chnbox, nax1 - chnbox - 1)
        for k in iterable:
            # prep filenames
            if negative:
                name_mask_tmp = 'mask_kernel' + str(minwidth) + '_I' + str(k) + '_negative'
            else:
                name_mask_tmp = 'mask_kernel' + str(minwidth) + '_I' + str(k) + '_positive'

            tmp_fits = os.path.join(tmpdir, name_mask_tmp + '.fits')
            tmp_list = os.path.join(tmpdir, 'target_' + name_mask_tmp + '.list')

            # collapsing cube over chosen channel number, saving rms in center
            im_channel_sum = np.nansum(cube[k - chnbox:k + chnbox + 1, :, :], axis=0)
            rms = np.nanstd(
                im_channel_sum[int(nax2 / 2) - int(nax2 * rms_region) - 1:int(nax2 / 2) + int(nax2 * rms_region),
                int(nax3 / 2) - int(nax3 * rms_region) - 1:int(nax3 / 2) + int(nax3 * rms_region)])

            # create fitsfile
            hdu = fits.PrimaryHDU(data=im_channel_sum, header=self.head)
            hdul = fits.HDUList([hdu])
            hdul.writeto(tmp_fits, overwrite=True)

            # run Sextractor
            path_sextractor_files = os.path.dirname(interferopy.__file__ ) + '/data/'

            if sextractor_param_file=='':
                sextractor_param_file = path_sextractor_files + 'default.sex'

            os.system('sex ' + tmp_fits + ' -c ' + sextractor_param_file
                      + ' -PARAMETERS_NAME '+ path_sextractor_files + 'default.param'
                      + ' -CATALOG_NAME ' + tmp_list + ' -VERBOSE_TYPE QUIET')

            # read output
            sextractor_cat = np.genfromtxt(tmp_list, skip_header=6)

            # process output
            if sextractor_cat.shape == (0,):
                if clean_tmp:
                    os.remove(tmp_fits)
                    os.remove(tmp_list)
                continue
            elif len(sextractor_cat.shape) == 1:
                sextractor_cat = sextractor_cat.reshape((-1, 6))

            # append k , rms, freq to sextractor_cat
            sextractor_cat = np.hstack([sextractor_cat, np.ones((len(sextractor_cat), 1)) * k,
                                        np.ones((len(sextractor_cat), 1)) * rms,
                                        np.ones((len(sextractor_cat), 1)) * self.freqs[k]])
            if k == chnbox:
                np.savetxt(fname=output_file + '_kw' + str(int(minwidth)) + '.cat', X=sextractor_cat,
                           header="# 1 SNR_WIN                Gaussian-weighted SNR \n \
                           #   2 FLUX_MAX               Peak flux above background                                 [count] \n \
                           #   3 X_IMAGE                Object position along x                                    [pixel] \n \
                           #   4 Y_IMAGE                Object position along y                                    [pixel] \n \
                           #   5 ALPHA_J2000            Right ascension of barycenter (J2000)                      [deg] \n \
                           #   6 DELTA_J2000            Declination of barycenter (J2000)                          [deg] \n \
                           #   7 k                      Central Channel                                            [pixel] \n \
                           #   8 RMS                    RMS of collapsed channel map                               [Jy/beam] \n \
                           #   9 FREQ                   CENTRAL FREQUENCY [GHz]                                         [Hz] ")
            else:
                with open(output_file + '_kw' + str(int(minwidth)) + '.cat', "ab") as f:
                    np.savetxt(fname=f, X=sextractor_cat)

            # cleanup intermediate files from this loop
            if clean_tmp:
                os.remove(tmp_fits)
                os.remove(tmp_list)

        # finally remove tmpdir
        if clean_tmp:
            os.rmdir(tmpdir)

    def findclumps_full(self, output_file,
                        kernels=np.arange(3, 20, 2), rms_region=1./4.,
                        sextractor_param_file='',
                        clean_tmp=True, ncores=1,
                        run_search=True, run_positive=True, run_negative=True,
                        run_crop=True,
                        SNR_min=3, delta_offset_arcsec=2, delta_freq=0.1,
                        run_fidelity=True,
                        fidelity_bins=np.arange(0, 15, 0.2),
                        min_SN_fit=4.0, fidelity_threshold=0.5,
                        verbose=True):
        '''Run the full findclumps search and analysis for different kernel
        sizes, on the positive and/or negative cube(s):

        1. Search (see `findclumps_1kernel()`)
        2. Crop doubles and trim candidates above `min_SNR` (see `tools.run_line_stats_sex` and `tools_crop_doubles`)
        3. Determine the fidelity and produce catalog of candidates (see `tools.fidelity_analysis`)

        To skip any of the steps, use the appropriate flags.  This is
        intended to be used to re-run parts of the search and/or
        analysis after changing the parameters (e.g., adding different
        kernels and/or modifying the cropping criteria).

        :param output_file: relative/absolute path to the output catalogue
        :param kernels: list of odd kernel widths to use
        :param rms_region: Region to compute the rms noise [2x2 array in image pixel coord].
            If ``None``, takes the central 25% pixels (square)
        :param sextractor_param_file: path to sextractor param file
        :param clean_tmp: cleanup intermediate sextractor files (disable for debugging)
        :param ncores: number of cores for multiprocessing (done over kernels and pos/neg search)
        :param run_search: if `False` will skip the search step
        :param run_positive: if `False` will skip the positive search
        :param run_negative: if `False` will skip the negative search

        :param run_crop: if `False` will skip the crop step
        :param SNR_min: Minimum SN of peaks to retain in the crop
        :param delta_offset_arcsec: maximum offset to match detections in the cube [arcsec]
        :param delta_freq: maximum frequency offset to match detections in the cube [GHz]

        :param run_fidelity: if `False` will skip the fidelity analysis
        :param fidelity_bins: SN bins for the fidelity analysis
        :param min_SN_fit: minimum SN for fidelity fit
        :param fidelity_threshold: Fidelity threshold above which to select candidates

        :param verbose: increase overall verbosity
        '''
        if run_search:
            if ncores == 1:
                for i in kernels:
                    if run_positive:
                        self.findclumps_1kernel(output_file=output_file + '_clumpsP', negative=False, minwidth=i,
                                                clean_tmp=clean_tmp, rms_region=rms_region,
                                                sextractor_param_file=sextractor_param_file,
                                                verbose=verbose)
                    if run_negative:
                        self.findclumps_1kernel(output_file=output_file + '_clumpsN', negative=True, minwidth=i,
                                                clean_tmp=clean_tmp, rms_region=rms_region,
                                                sextractor_param_file=sextractor_param_file,
                                                verbose=verbose)

            else:
                from multiprocessing.dummy import Pool as ThreadPool
                from itertools import repeat

                kernels = np.atleast_1d(kernels)
                if run_positive and run_negative:
                    kernels_width_neg_and_pos = np.concatenate((-kernels, kernels))
                else:
                    if run_positive:
                        kernels_width_neg_and_pos = kernels
                    elif run_negative:
                        kernels_width_neg_and_pos = -kernels
                    else:
                        # this should not happen
                        raise RuntimeError("ERROR: run_search=True but run_positve=run_negative=False; "
                                           "NotImplemented for Multiprocessing: set run_search=False")
                names = [output_file + '_clumpsN'] * len(kernels) + [output_file + '_clumpsP'] * len(kernels)
                # arguments: (output_file, rms_region, minwidth, sextractor_param_file, clean_tmp, negative)
                iterable = zip(names, repeat(rms_region), np.abs(kernels_width_neg_and_pos),
                               repeat(sextractor_param_file), repeat(clean_tmp), (np.sign(kernels_width_neg_and_pos) < 0),
                               repeat(verbose))

                with ThreadPool(ncores) as p:
                    p.starmap(self.findclumps_1kernel, iterable)
                    p.close()
                    p.join()
            if verbose:
                print('Findclumps done.')
            
        if run_crop:
            if verbose:
                print('Running line stats and cropping doubles...')
            
            # process positive catalog
            if run_positive:
                tools.run_line_stats_sex(sextractor_catalogue_name=output_file + '_clumpsP',
                                         binning_array=kernels, SNR_min=SNR_min)

                tools.crop_doubles(cat_name=output_file + "_clumpsP_minSNR_" + str(SNR_min) + ".cat",
                                   delta_offset_arcsec=delta_offset_arcsec,
                                   delta_freq=delta_freq,
                                   verbose=verbose)

            # process negative catalog
            if run_negative:
                tools.run_line_stats_sex(sextractor_catalogue_name=output_file + '_clumpsN',
                                         binning_array=kernels, SNR_min=SNR_min)

                tools.crop_doubles(cat_name=output_file + "_clumpsN_minSNR_" + str(SNR_min) + ".cat",
                                   delta_offset_arcsec=delta_offset_arcsec,
                                   delta_freq=delta_freq,
                                   verbose=verbose)

        # analyse fidelity
        if run_fidelity:
            catP, catN, candP, candN \
                = tools.fidelity_analysis(catN_name=output_file + "_clumpsN_minSNR_" + str(SNR_min) + "_cropped.cat",
                                          catP_name=output_file + "_clumpsP_minSNR_" + str(SNR_min) + "_cropped.cat",
                                          bins=fidelity_bins,
                                          min_SN_fit=min_SN_fit,
                                          fidelity_threshold=fidelity_threshold,
                                          kernels=kernels, verbose=verbose)

            return catP, catN, candP, candN


class MultiCube:
    """
    A container like object to hold multiple cubes at the same time. Cubes are stored in a dictionary.
    Allows performing tasks such as residual scaled aperture integration or spectrum extraction.
    Load another cube into MultiCube object by e.g. mc.load_cube("path_to_cube.residual.fits", "residual")

    :Example:

    .. code-block:: python

        filename="cube.fits"
        ra, dec, freq = (205.533741, 9.477317341, 222.54) # coord for the emission line
        mc=MultiCube(filename) # will automatically try to load cube.xxx.fits, where xxx is residual, dirty, pb, model, psf, or image.pbcor

        # Alternatively load each cube manually
        # mc = MultiCube()
        # mc.load_cube("somewhere/cube.fits", "image")
        # mc.load_cube("elsewhere/cube.dirty.fits", "dirty")
        # mc.load_cube("elsewhere/cube.residual.fits", "residual")
        # mc.load_cube("elsewhere/cube.pb.fits", "pb")

        # spectrum extracted from the circular aperture, with associated error, corrected for residual and PB response
        flux, err, tab = mc.spectrum_corrected(ra=ra, dec=dec, radius=1.5, calc_error=True)
        # tab.write("spectrum.txt", format="ascii.fixed_width", overwrite=True)  # Save results for later

        # curve of growth up to the maximum radius, in steps of one pixel, in a chosen frequency channel
        radius, flux, err, tab = mc.growing_aperture_corrected(ra=ra, dec=dec, freq=freq, maxradius=5, calc_error=True)
        # tab.write("growth.txt", format="ascii.fixed_width", overwrite=True)  # Save results for later
    """

    def __init__(self, filename: str = None, autoload_multi=True):
        """
        Provide the file path to the final cleaned cube. Will try to find other adjacent cubes based on their names.
        Standard key names from CASA are: image, residual, model, psf, pb, image.pbcor
        Additional names are dirty, clean.comp

        :param filename: Path string to the cleaned cube fits image.
        :param autoload_multi: If true, attempt to find other cubes using preset (mostly CASA) suffixes.
        """

        # these are standard suffixes from CASA tclean output (except "dirty")
        # gildas output has different naming conventions, which are not implemented here
        keylist = ["image", "residual", "dirty", "pb", "model", "psf", "image.pbcor", "clean.comp"]
        self.cubes = dict(zip(keylist, [None] * len(keylist)))
        self.basename = None

        if filename is None:
            # just setup the basic dictionary and exit
            return
        elif not os.path.exists(filename):
            raise FileNotFoundError(filename)

        filenames = dict(zip(keylist, [None] * len(keylist)))
        filenames["image"] = filename

        # TODO could improve searching, but there are infinite number of ways people could name these files
        # get the basename, which is hopefully shared between different cubes
        extension = ".fits"
        endings = [".image.tt0.fits", ".image.fits", ".fits"]
        for ending in endings:
            if filename.lower().endswith(ending):
                self.basename = filename[:-len(ending)]
                extension = ending.replace(".image", "")
                break

        # try to fill other filenames
        if autoload_multi:
            if self.basename is None:
                raise ValueError("Unknown extension.")

            # print("extension",extension)
            for k in keylist:
                # expected filename
                filename_suffixed = self.basename + "." + k + extension
                if os.path.exists(filename_suffixed):
                    filenames[k] = filename_suffixed

        # load the cubes
        for k in self.cubes.keys():
            if filenames[k] is not None:
                self.cubes[k] = Cube(filenames[k])

        self.log("Loaded cubes: " + str(self.loaded_cubes))

    def load_cube(self, filename: str, key: str = None):
        """
        Add provided fits file to the MultiCube instance.
        Known keys are: "image", "residual", "dirty", "pb", "model", "psf", "image.pbcor"
        If the filename does not end with these words, please provide a manual key.

        :param filename: Path string to the fits image.
        :param key: Dictionary key name used to store the cube.
        :return:
        """
        # try to estimate the key from the filename
        if key is None:
            # assuming the file ends in something like .residual.fits
            suffix = str(filename).split(".")[-2]
            # if this suffix matches preset ones
            if suffix in self.cubes.keys():
                key = suffix

        # Load and store new cube
        cub = Cube(filename)

        if key is None:
            raise ValueError("Please provide key under which to store the cube.")

        if len(self.loaded_cubes) > 0:
            if cub.im.shape != self[self.loaded_cubes[0]].im.shape:
                self.log("Warning: Loaded cube has a different shape than existing one!")

        self[key] = cub

    def __getitem__(self, key: str):
        return self.cubes[key]

    def __setitem__(self, key: str, value: Cube):
        self.cubes[key] = value

    def get_loaded_cubes(self):
        """
        Get a list of loaded cubes (those that are not None).

        :return: List of key names.
        """
        return [k for k in self.cubes.keys() if self.cubes[k] is not None]

    loaded_cubes = property(get_loaded_cubes)
    """List of keys corresponding to loaded cubes."""

    def get_freqs(self):
        if len(self.loaded_cubes) > 0:
            return self[self.loaded_cubes[0]].freqs  # maybe pick explicitely self["image"].freqs
        else:
            raise ValueError("No cubes present!")

    freqs = property(get_freqs)
    """Array of frequencies corresponding to cube channels (from the first loaded cube)."""

    def log(self, text: str):
        """
        Basic logger function to allow better functionality in the future development.
        All class functions print info through this wrapper.
        Could be extended to provide different levels of info, timestamps, or logging to a file.
        """
        print(text)

    def make_clean_comp(self, overwrite=False):
        """
        Generate a clean component cube. Defined as the cleaned cube minus the residual, or, alternatively,
        model image convolved with the clean beam. This cube is not outputted by CASA.

        :param overwrite: If true, overrides any present "clean.comp" cube.
        :return: None
        """
        self.log("Generate clean component cube.")

        if self["image"] is None:
            raise ValueError("Cleaned cube is missing.")
        if self["residual"] is None:
            raise ValueError("Residual cube is missing.")
        if self["clean.comp"] is not None and not overwrite:
            self.log("Warning: clean.comp cube already exists and overwriting is disabled.")

        cub = Cube(self["image"].filename)
        cub.im = self["image"].im - self["residual"].im
        cub.filename = None  # This cube is no longer the one on disk, so empty the filename as a precaution
        self["clean.comp"] = cub

    # return self["clean.comp"]

    def make_flatnoise(self, overwrite=False):
        """
        Generate a flat noise cube from the primary beam (PB) corrected one, and the PB response (which is <= 1).
        PB corrected cube has valid fluxes, but rms computation is not straightforward.
        PB corrected cube = flat noise cube / PB response

        :param overwrite: If true, overrides any present "image"" cube.
        :return: None
        """
        self.log("Generate flat noise image cube from the PB corrected one.")

        if self["image.pbcor"] is None:
            raise ValueError("PB corrected cleaned cube is missing.")
        if self["pb"] is None:
            raise ValueError("PB response cube is missing.")
        if self["image"] is not None and not overwrite:
            self.log("Warning: image cube already exists and overwriting is disabled.")

        cub = Cube(self["image.pbcor"].filename)
        cub.im = self["image.pbcor"].im * self["pb"].im
        cub.filename = None  # This cube is no longer the one on disk, so empty the filename as a precaution
        self["image"] = cub

    # return self["image"]

    def __cubes_prepare(self) -> bool:
        """
        Perform several prelimiaries before attempting residual scaled flux extraction.

        :return: True if no problems are detected, False otherwise.
        """

        # if necessary, use pb corrected map and pb to generate flat noise map
        only_pbcor_exists = self["image"] is None and self["image.pbcor"] is not None and self["image.pb"] is not None
        if only_pbcor_exists:
            self.make_flatnoise()

        cubes_exists = self["image"] is not None and self["dirty"] is not None and self["residual"] is not None
        if not cubes_exists:
            self.log("Error: Need all three cubes: image, dirty, residual!")
            return False

        # generate clean component cube (this is not a standard CASA output)
        # actually it is not necessary to have the full cube for residual scaling
        # if self["clean.comp"] is None:
        # 	self.make_clean_comp()

        shapes_equal = self["image"].im.shape == self["dirty"].im.shape == self["residual"].im.shape
        if not shapes_equal:
            self.log("Error: Cube shapes are not equal!")
            return False

        # Residual scaling assumes clean beam in all maps
        # Override the beam volume in dirty and residual maps to the cleaned one
        bvol = self["image"].beamvol
        self["dirty"].beamvol = bvol
        self["residual"].beamvol = bvol

        return True

    def spectrum_corrected(self, ra: float = None, dec: float = None, freq: float = None, radius: float = 1.0,
                           px: int = None, py: int = None, channel: int = None,
                           calc_error=True, sn_cut: float = 2.5, apply_pb_corr=True):
        """
        Extract aperture integrated spectrum from the cube using the residual scaling to account for the dirty beam.
        Correction for the primary beam response is applied if avaliable.
        Coordinates can be given in degrees (ra, dec) or pixels (px, py).
        If no coordinates are given, the center of the map is assumed.

        For details on the method see appendix A in Novak et al. (2019):
        https://ui.adsabs.harvard.edu/abs/2019ApJ...881...63N/abstract

        :param ra: Right ascention in degrees.
        :param dec: Declination in degrees.
        :param radius: Circular aperture radius in arcsec.
        :param px: Right ascention pixel coord (alternative to ra).
        :param py: Declination pixel coord (alternative to dec).
        :param channel: Channel index (alternative to freq)
        :param freq: Frequency in GHz. Extract only in a single channel instead of the full cube.
        :param calc_error: Set to False to skip error calculations, if the rms computation is slow or not necessary.
        :param sn_cut: Use emission above this S/N threshold to estimate the clean-to-dirty beam ratio, need calc_error.
        :param apply_pb_corr: Scale flux and error by the primary beam response (single pix value), needs loaded pb map.
        :return: flux, err, tab: 1D array for corrected flux and error estimate; tab is a Table with all computations.
        """

        # take single pixel value if no aperture radius given
        if radius <= 0:
            self.log('No aperture defined - taking a single pixel')
        # raise ValueError("No aperture is defined!")

        # check and prepare cubes for residual scaling
        if not self.__cubes_prepare():
            raise ValueError("Cubes check failed!")

        if freq is not None:
            channel = self["image"].freq2pix(freq)

        # run aperture extraction on all cubes
        # the beam volume should be the same in all of them (checked by __cubes_prepare)
        params = dict(ra=ra, dec=dec, radius=radius, px=px, py=py, channel=channel, freq=freq)
        flux_image, err, npix, peak_sb = self["image"].spectrum(calc_error=calc_error,
                                                                **params)  # compute rms only on this map
        flux_residual, _, _, _ = self["residual"].spectrum(calc_error=False, **params)
        flux_dirty, _, _, _ = self["dirty"].spectrum(calc_error=False, **params)
        flux_clean = flux_image - flux_residual

        # alternatively, force creation of the clean components cube
        # self.make_clean_comp()
        # flux_clean, _, _ = self["clean.comp"].spectrum(calc_error=False, **params)

        # this can be numerically unstable if there is no clean flux or if dirty == residual
        # nothing much to do about it, it's a limitation of the method
        epsilon = flux_clean / (flux_dirty - flux_residual)

        # estimate a single epsilon across the full spectrum
        # it is not expected to change a lot across channels (if PSF is consistent, and bandwidth is not too large)
        epsilon_fix = float(np.nanmedian(epsilon))
        if calc_error:
            if channel is None:
                rmses = self["image"].rms
            else:
                rmses = np.array([self["image"].rms[channel]])
            # use only high S/N channels
            w = (flux_clean / err) > sn_cut
            if np.sum(w) > 0:
                epsilon_fix = np.nanmedian(epsilon[w])
            else:
                self.log("Warning: Nothing above S/N>" + str(sn_cut) + ", using all channels for epsilon.")
        else:
            rmses = np.full_like(flux_image, fill_value=np.nan)
            self.log("Using all channels for clean-to-dirty beam ratio (epsilon).")

        # corrected flux is estimated from the dirty map and the clean-to-dirty beam ratio (epsilon)
        flux = epsilon_fix * flux_dirty
        err = epsilon_fix * err
        peak_sb = peak_sb

        # apply PB correction if possible
        if apply_pb_corr and "pb" in self.loaded_cubes:
            self.log("Correcting flux and err for PB response.")
            pb, _, _, _ = self["pb"].spectrum(ra=ra, dec=dec, px=px, py=py, channel=channel, freq=freq,
                                              calc_error=False)
            flux = flux / pb
            err = err / pb
            peak_sb = peak_sb / pb

        elif apply_pb_corr and "pb" not in self.loaded_cubes:
            self.log("Warning: Cannot correct for PB, missing pb map.")
            pb = np.full(len(flux_image), np.nan)
        else:
            self.log("Flux is not PB corrected.")
            pb = np.full(len(flux_image), np.nan)

        # crude estimate if something went poorly, epsilon is too small or too big
        if epsilon_fix < 0.01 or epsilon_fix > 100:
            self.log("Warning: The clean-to-dirty beam ratio (epsilon) is badly determined.")
        epsilon_fix = np.full(len(flux), epsilon_fix)

        if channel is None:
            channels = np.arange(len(flux))
        else:
            channels = [channel]

        freqs = self.freqs[channels]
        nbeam = npix / self["image"].beamvol

        tab = Table([channels,  # Channel index
                     freqs,  # Frequency of the channel in GHz
                     flux,  # Aperture flux corrected for residual and primary beam response
                     err,  # Error estimate on the corrected flux
                     epsilon_fix,  # Best estimate of the clean-to-dirty beam ratio, fixed across all channels
                     flux_image,  # Aperture flux measured in the final map (assumes clean beam)
                     flux_dirty,  # Aperture flux measured in the dirty map (assumes clean beam)
                     flux_residual,  # Aperture flux measured in the residual map (assumes clean beam)
                     flux_clean,  # Aperture flux measured in the clean components map (= final - residual)
                     npix,  # Number of pixels inside the aperture
                     nbeam,  # Number of beams inside the aperture
                     epsilon,  # Clean-to-dirty beam ratio in the channel
                     rmses,  # Rms noise of the channel
                     peak_sb,  # (Corrected) Peak Surface Brightness of the channel
                     pb],  # Primary beam response (<=1) at the given coordinate, single pixel value
                    names=["channel", "freq", "flux", "err", "epsilon_fix",
                           "flux_image", "flux_dirty", "flux_residual", "flux_clean",
                           "npix", "nbeam", "epsilon", "rms", "peak_sb", "pb"])

        return np.array(flux), np.array(err), tab

    def growing_aperture_corrected(self, ra: float = None, dec: float = None, freq: float = None,
                                   maxradius=1.0, binspacing: float = None, bins: list = None,
                                   px: int = None, py: int = None, channel: int = 0,
                                   calc_error=True, apply_pb_corr=True):
        """
        Extract the curve of growth from the map using the residual scaling to account for the dirty beam.
        Correction for the primary beam response is applied if avaliable.
        Coordinates can be given in degrees (ra, dec) or pixels (px, py).
        If no coordinates are given, the center of the map is assumed.

        For details on the method see appendix A in Novak et al. (2019):
        https://ui.adsabs.harvard.edu/abs/2019ApJ...881...63N/abstract

        :param ra: Right ascention in degrees.
        :param dec: Declination in degrees.
        :param freq: Frequency in GHz.
        :param maxradius: Max radius for aperture integration in arcsec.
        :param binspacing: Resolution of the growth flux curve in arcsec, default is one pixel size.
        :param bins: Custom bins for curve growth (1D np array).
        :param px: Right ascention pixel coord (alternative to ra).
        :param py: Declination pixel coord (alternative to dec).
        :param channel: Index of the cube channel to take (alternative to freq). Default is the first channel.
        :param calc_error: Set to False to skip error calculations, if the rms computation is slow or not necessary.
        :param apply_pb_corr: Scale flux and error by the primary beam response (single pix value), needs loaded pb map.
        :return: radius, flux, err, tab:  1D array for radius[arcsec] and corrected flux and error estimate;
            tab is a Table with all computations.
        """

        # check and prepare cubes for residual scaling
        if not self.__cubes_prepare():
            raise ValueError("Cubes check failed!")

        # define channel to use
        if freq is not None:
            channel = self["image"].freq2pix(freq)

        # run growing aperture extraction on all cubes in a single channel
        # the beam volume should be the same in all of them (checked by __cubes_prepare)
        params = dict(ra=ra, dec=dec, maxradius=maxradius,
                      px=px, py=py, channel=channel, freq=freq,
                      binspacing=binspacing, bins=bins, profile=False)  # shared parameters
        radius, flux_image, err, npix = self["image"].growing_aperture(calc_error=calc_error, **params)  # rms here only
        _, flux_residual, _, _ = self["residual"].growing_aperture(calc_error=False, **params)
        _, flux_dirty, _, _ = self["dirty"].growing_aperture(calc_error=False, **params)
        flux_clean = flux_image - flux_residual

        epsilon = flux_clean / (flux_dirty - flux_residual)
        flux = np.array(epsilon * flux_dirty)
        err = epsilon * err
        nbeam = npix / self["image"].beamvol[channel]

        # apply PB correction if possible
        if apply_pb_corr and "pb" in self.loaded_cubes:
            self.log("Correcting flux and err for PB response.")
            # will return a single pb value, because channel is set
            pb, _, _, _ = self["pb"].spectrum(ra=ra, dec=dec, px=px, py=py, channel=channel, freq=freq,
                                              calc_error=False)
            pb = np.full(len(flux_image), pb)
            flux = np.array(flux / pb)
            err = np.array(err / pb)
        elif apply_pb_corr and "pb" not in self.loaded_cubes:
            self.log("Warning: Cannot correct for PB, missing pb map.")
            pb = np.full(len(flux_image), np.nan)
        else:
            self.log("Flux is not PB corrected.")
            pb = np.full(len(flux_image), np.nan)

        tab = Table([radius,  # Aperture radius in arcsec.
                     flux,  # Aperture flux corrected for residual
                     err,  # Error estimate on the corrected flux
                     flux_image,  # Aperture flux measured in the final map (assumes clean beam)
                     flux_dirty,  # Aperture flux measured in the dirty map (assumes clean beam)
                     flux_residual,  # Aperture flux measured in the residual map (assumes clean beam)
                     flux_clean,  # Aperture flux measured in the clean components map (= final - residual)
                     epsilon,  # Clean-to-dirty beam ratio for a given aperture size
                     npix,  # Number of pixels inside the aperture
                     nbeam,  # Number of beams inside the aperture: nbeam = npix / beamvol
                     pb],  # Primary beam response (<=1) applied to flux and err columns, single value at all radii
                    names=["radius", "flux", "err",
                           "flux_image", "flux_dirty", "flux_residual", "flux_clean",
                           "epsilon", "npix", "nbeam", "pb"])

        return radius, flux, err, tab
