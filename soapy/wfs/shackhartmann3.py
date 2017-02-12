import numpy
import numpy.random
import pyfftw

try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("Soapy requires either pyfits or astropy")

from .. import AOFFT, LGS, logger
from .. import aotools
from ..aotools import centroiders, wfs, interp, circle
from .. import numbalib

# xrange now just "range" in python3.
# Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32


ASEC2RAD = (numpy.pi/(180 * 3600))

class ShackHartmann3(object):
    """
    A Shack-Hartmann WFS

    An object that accepts a phase instance, and calculates the resulting measurements as would be observed by a Shack-Hartmann WFS. To do this, the phase is first interpolated linearly to make a size that will return the desired pixel scale. The phase is then split into sub-apertures, each of which is brought to a focal plane using an FFT. The individual sub-apertures are placed back into a SH WFS detector image. A centroiding algorithm is then performed on each sub-aperture this detector array

    Parameters:
        soapy_config (SoapyConfig):
        wfs_index (int):
        mask (ndarray):
        los (LineOfSight, optional): Corresponding Soapy Line of sight object. Can be given so its easier for other modules to retrieve it

    """
    def __init__(self, soapy_config, n_wfs, mask=None, los=None):

        # Sort out the paramters teh WFS will need (that can't be changed dynamically)
        # --------------------------------
        self.los = los

        self.soapy_config = soapy_config
        self.config = soapy_config.wfss[n_wfs]

        # Sim params
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.telescope_diameter = self.soapy_config.tel.telDiam
        self.threads = self.soapy_config.sim.threads
        self.phase_pixel_scale = 1./self.soapy_config.sim.pxlScale

        # WFS Params
        self.gs_mag = self.config.GSMag
        self.subap_fov = self.config.subapFOV
        self.wavelength = self.config.wavelength
        self.nx_subaps = self.config.nxSubaps
        self.subap_threshold = self.config.subapThreshold
        self.nx_subap_pixels = self.config.pxlsPerSubap
        self.fft_oversamp = self.config.fftOversamp
        self.lgs_config = self.config.lgs

        # Calculate some parameters
        # -------------------------
        self.subap_diam = self.telescope_diameter / self.nx_subaps

        # spacing between subaps in pupil Plane (size "pupil_size")
        self.nx_subap_pupil = float(self.pupil_size)/self.nx_subaps

        # Spacing on the "FOV Plane" - the number of elements required
        # for the correct subap FOV (from way FFT "phase" to "image" works)
        self.subap_fov_rad = self.subap_fov * ASEC2RAD
        self.nx_subap_interp = int(round(
                self.subap_diam * self.subap_fov_rad/ self.wavelength))

        # make subap twice as big to double subap FOV
        if self.config.subapFieldStop==True:
            self.SUBAP_OVERSIZE = 1
        else:
            self.SUBAP_OVERSIZE = 2

        self.nx_detector_pixels = self.nx_subap_pixels * self.config.nxSubaps
        self.nx_subap_interp *= self.SUBAP_OVERSIZE
        self.nx_subap_pixels2 = (self.SUBAP_OVERSIZE * self.nx_subap_pixels)

        # The total size of the required EField for all subaps.
        self.nx_interp_efield = int(round(
                self.nx_subaps * self.nx_subap_interp))

        # Calculate the subaps that are actually seen behind the pupil mask
        self.pupil_subap_coords, self.subap_fill_factor = wfs.findActiveSubaps(
                self.nx_subaps, mask, self.subap_threshold, returnFill=True)
        self.interp_subap_coords = numpy.round(
                self.pupil_subap_coords * self.nx_subap_interp / self.nx_subap_pupil)

        self.n_subaps = int(self.pupil_subap_coords.shape[0])
        self.n_measurements = 2 * self.n_subaps

        # Coordinates to put the subaps into the detector
        self.detector_subap_coords = numpy.round(
                self.pupil_subap_coords*(self.nx_detector_pixels/float(self.pupil_size)))

        # Calculate the FFT padding to use
        self.subapFFTPadding = self.nx_subap_pixels2 * self.fft_oversamp
        if self.subapFFTPadding < self.nx_subap_interp:
            while self.subapFFTPadding<self.nx_subap_interp:
                self.fft_oversamp+=1
                self.subapFFTPadding\
                        =self.nx_subap_pixels2*self.fft_oversamp

            logger.warning("requested WFS FFT Padding less than FOV size... Setting oversampling to: %d"%self.fft_oversamp)

        # Init the FFT to the focal plane
        # self.FFT = AOFFT.FFT(
        #         inputSize=(
        #         self.n_subaps, self.subapFFTPadding, self.subapFFTPadding),
        #         axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
        #         THREADS=self.threads,
        #         fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT"))

        self.subap_interp_efield = numpy.zeros(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
        self.binnedFPSubapArrays = numpy.zeros(
                (self.n_subaps, self.nx_subap_pixels2, self.nx_subap_pixels2), dtype=DTYPE)
        self.subap_focus_efield = numpy.zeros(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=CDTYPE)
        self.subap_focus_intensity = numpy.zeros(
                (self.n_subaps, self.subapFFTPadding, self.subapFFTPadding), dtype=DTYPE)

        self.detector = numpy.zeros(
                (self.nx_detector_pixels, self.nx_detector_pixels), dtype=DTYPE)
        # Array used when centroiding subaps
        self.centSubapArrays = numpy.zeros( (self.n_subaps,
              self.nx_subap_pixels, self.nx_subap_pixels) )

        self.fft = pyfftw.FFTW(
                self.subap_interp_efield, self.subap_focus_efield,
                threads=self.threads, axes=(1,2))

        # If LGS uplink, init FFTs to conovolve LGS PSF and WFS PSF(s)
        # This works even if no lgs_config.uplink as ``and`` short circuits
        if self.lgs_config and self.lgs_config.uplink:
            self.iFFT = AOFFT.FFT(
                    inputSize = (self.n_subaps,
                                        self.subapFFTPadding,
                                        self.subapFFTPadding),
                    axes=(-2,-1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=self.threads,
                    fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT")
                    )

            self.lgs_iFFT = AOFFT.FFT(
                    inputSize = (self.subapFFTPadding,
                                self.subapFFTPadding),
                    axes=(0,1), mode="pyfftw",dtype=CDTYPE,
                    THREADS=self.threads,
                    fftw_FLAGS=(self.config.fftwFlag,"FFTW_DESTROY_INPUT")
                    )


            if self.lgs_config.uplink:
                lgsObj = getattr(
                        LGS, "LGS_{}".format(self.lgs_config.propagationMode))
                self.lgs = lgsObj(
                        self.config, self.soapyConfig,
                        nOutPxls=self.subapFFTPadding,
                        outPxlScale=float(self.subap_fov)/self.subapFFTPadding
                        )



        self.slopes = numpy.zeros(self.n_measurements)

        self.setMask(mask)
        self.calculate_tilt_correction()

        # Make flat wavefront, and run through WFS in iMat mode to turn off features
        phs = numpy.zeros([self.pupil_size]*2).astype(DTYPE)
        self.reference_slopes = self.slopes.copy()
        self.reference_slopes = self.frame(phs).copy()

    def setMask(self, mask):

        if numpy.any(mask):
            self.mask = mask
        else:
            self.mask = circle.circle(
                    self.pupil_size/2., self.pupil_size,
                    )
        # Find the mask to apply to the scaled EField
        self.scaled_mask = numpy.round(interp.zoom(
                    self.mask, self.nx_interp_efield))

        self.subap_fill_factor = wfs.computeFillFactor(
                mask, self.pupil_subap_coords, self.nx_subap_pupil)


    def calculate_tilt_correction(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel
        """
        if not self.nx_subap_pixels%2:
            # If pxlsPerSubap is even
            # Angle we need to correct for half a pixel
            theta = self.SUBAP_OVERSIZE*self.subap_fov_rad/ (
                    2*self.subapFFTPadding)

            # Magnitude of tilt required to get that angle
            A = theta * self.subap_diam/(2*self.wavelength)*2*numpy.pi

            # Create tilt arrays and apply magnitude
            coords = numpy.linspace(-1, 1, self.nx_subap_interp)
            X,Y = numpy.meshgrid(coords,coords)

            self.tilt_fix = -1 * A * (X+Y)

        else:
            self.tilt_fix = numpy.zeros((self.nx_subap_interp,)*2)


    def zeroData(self, detector=True, FP=True):
        """
        Sets data structures in WFS to zero.

        Parameters:
            detector (bool, optional): Zero the detector? default:True
            FP (bool, optional): Zero intermediate focal plane arrays? default: True
        """

        self.zeroPhaseData()

        if FP:
            self.FPSubapArrays[:] = 0

        if detector:
            self.detector[:] = 0

    def interp_to_size(self):
        # Have to make phase the correct size for pixel scale
        self.scaledEField = interp.zoom(self.phase, self.nx_interp_efield)
        self.scaledEField = numpy.exp(1j*self.scaledEField)

        # Apply the scaled pupil mask
        self.scaledEField *= self.scaled_mask

    def chop_into_subaps(self):

        #
        # numbalib.wfs.chop_subaps(
        #         self.scaledEField, self.interp_subap_coords, self.nx_subap_interp,
        #         self.subap_interp_efield, threads=self.threads)


        # create an array of individual subap EFields
        for i in xrange(self.n_subaps):
            x, y = numpy.round(self.pupil_subap_coords[i] *
                                     self.nx_subap_interp/self.nx_subap_pupil)
            self.subap_interp_efield[i,
                    :self.nx_subap_interp,
                    :self.nx_subap_interp
                    ] = self.scaledEField[
                                    int(x):
                                    int(x+self.nx_subap_interp) ,
                                    int(y):
                                    int(y+self.nx_subap_interp)] * numpy.exp(1j * self.tilt_fix)

    def subaps_to_focus(self, intensity=1):
        #do the fft to all subaps at the same time
        # and convert into intensity®
        self.fft()

        self.subap_focus_intensity = intensity * numpy.abs(AOFFT.ftShift2d(self.subap_focus_efield))**2

        # if intensity==1:
        #     self.FPSubapArrays += numpy.abs(AOFFT.ftShift2d(self.subap_focus_intensity))**2
        # else:
        #     self.FPSubapArrays += intensity*numpy.abs(
        #             AOFFT.ftShift2d(self.subap_focus_intensity))**2

    def assemble_detector_image(self):
        # If required, convolve with LGS PSF
        if self.config.lgs and self.lgs and self.lgs_config.uplink and self.iMat != True:
            self.applyLgsUplink()

        # bins back down to correct size and then
        # fits them back in to a focal plane array
        self.binnedFPSubapArrays[:] = interp.binImgs(self.subap_focus_intensity,
                                                     self.fft_oversamp)

        # In case of empty sub-aps, will get NaNs
        self.binnedFPSubapArrays[numpy.isnan(self.binnedFPSubapArrays)] = 0

        # Scale each sub-ap flux by sub-aperture fill-factor
        self.binnedFPSubapArrays \
            = (self.binnedFPSubapArrays.T * self.subap_fill_factor).T

        for i in xrange(self.n_subaps):
            x, y = self.detector_subap_coords[i]

            # Set default position to put arrays into (SUBAP_OVERSIZE FOV)
            x1 = int(round(
                x + self.nx_subap_pixels / 2.
                - self.nx_subap_pixels2 / 2.))
            x2 = int(round(
                x + self.nx_subap_pixels / 2.
                + self.nx_subap_pixels2 / 2.))
            y1 = int(round(
                y + self.nx_subap_pixels / 2.
                - self.nx_subap_pixels2 / 2.))
            y2 = int(round(
                y + self.nx_subap_pixels / 2.
                + self.nx_subap_pixels2 / 2.))

            # Set defualt size of input array (i.e. all of it)
            x1_fp = int(0)
            x2_fp = int(round(self.nx_subap_pixels2))
            y1_fp = int(0)
            y2_fp = int(round(self.nx_subap_pixels2))

            # If at the edge of the field, may only fit a fraction in
            if x == 0:
                x1 = 0
                x1_fp = int(round(
                    self.nx_subap_pixels2 / 2.
                    - self.nx_subap_pixels / 2.))

            elif x == (self.nx_detector_pixels - self.nx_subap_pixels):
                x2 = int(round(self.nx_detector_pixels))
                x2_fp = int(round(
                    self.nx_subap_pixels2 / 2.
                    + self.nx_subap_pixels / 2.))

            if y == 0:
                y1 = 0
                y1_fp = int(round(
                    self.nx_subap_pixels2 / 2.
                    - self.nx_subap_pixels / 2.))

            elif y == (self.nx_detector_pixels - self.nx_subap_pixels):
                y2 = int(self.nx_detector_pixels)
                y2_fp = int(round(
                    self.nx_subap_pixels2 / 2.
                    + self.nx_subap_pixels / 2.))

            self.detector[x1:x2, y1:y2] += (
                self.binnedFPSubapArrays[i, x1_fp:x2_fp, y1_fp:y2_fp])

        # Scale data for correct number of photons
        self.detector /= self.detector.sum()
        self.detector *= aotools.photonsPerMag(
            self.gs_mag, self.mask, self.phase_pixel_scale,
            self.config.wvlBandWidth, self.config.exposureTime
        ) * self.config.throughput

        if self.config.photonNoise:
            self.addPhotonNoise()

        if self.config.eReadNoise != 0:
            self.addReadNoise()

    def calculate_slopes(self):
        # Sort out FP into subaps
        for i in xrange(self.n_subaps):
            x, y = self.detector_subap_coords[i]
            x = int(x)
            y = int(y)
            self.centSubapArrays[i] = self.detector[
                                      x:x + self.nx_subap_pixels,
                                      y:y + self.nx_subap_pixels].astype(DTYPE)

        slopes = getattr(centroiders, self.config.centMethod)(
            self.centSubapArrays,
            threshold=self.config.centThreshold,
            ref=None
        )

        # shift slopes relative to subap centre and remove static offsets
        self.slopes[:] = slopes.reshape(self.n_subaps * 2)
        self.slopes -= self.nx_subap_pixels / 2.0
        self.slopes -= self.reference_slopes


        if self.config.removeTT == True:
            self.slopes[:self.n_subaps] -= self.slopes[:self.n_subaps].mean()
            self.slopes[self.n_subaps:] -= self.slopes[self.n_subaps:].mean()

        if self.config.angleEquivNoise and not self.iMat:
            pxlEquivNoise = (
                self.wfsConfig.angleEquivNoise *
                float(self.nx_subap_pixels)
                / self.wfsConfig.subapFOV)
            self.slopes += numpy.random.normal(
                0, pxlEquivNoise, 2 * self.n_subaps)

        return self.slopes


    def applyLgsUplink(self):
        '''
        A method to deal with convolving the LGS PSF
        with the subap focal plane.
        '''

        self.lgs.getLgsPsf(self.los.scrns)

        self.lgs_iFFT.inputData[:] = self.lgs.psf
        self.iFFTLGSPSF = self.lgs_iFFT()

        self.iFFT.inputData[:] = self.FPSubapArrays
        self.iFFTFPSubapsArray = self.iFFT()

        # Do convolution
        self.iFFTFPSubapsArray *= self.iFFTLGSPSF

        # back to Focal Plane.
        self.FFT.inputData[:] = self.iFFTFPSubapsArray
        self.FPSubapArrays[:] = AOFFT.ftShift2d(self.FFT()).real



    def frame(self, phase, read=False):

        self.phase = phase.copy()

        self.interp_to_size()

        self.chop_into_subaps()

        self.subaps_to_focus()

        self.assemble_detector_image()

        self.calculate_slopes()

        return self.slopes