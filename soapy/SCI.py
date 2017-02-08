# Copyright Durham University and Andrew Reeves
# 2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.

import numpy
import scipy.optimize as opt
import pyfftw

from . import AOFFT, logger, lineofsight
from .aotools import circle, interp
from . import numbalib

DTYPE = numpy.float32
CDTYPE = numpy.complex64


class PSF(object):

    def __init__(self, soapyConfig, nSci=0, mask=None):

        self.soapyConfig = soapyConfig
        self.simConfig = soapyConfig.sim
        self.telConfig = soapyConfig.tel
        self.config = self.sciConfig = soapyConfig.scis[nSci] # For compatability
        self.atmosConfig = soapyConfig.atmos
        self.mask = mask
        self.FOVrad = self.config.FOV * numpy.pi / (180. * 3600)

        self.FOVPxlNo = int(numpy.round(
            self.telConfig.telDiam * self.FOVrad
            / self.config.wavelength))

        self.padFOVPxlNo = int(round(
            self.FOVPxlNo * float(self.simConfig.simSize)
            / self.simConfig.pupilSize)
        )
        if self.padFOVPxlNo % 2 != self.FOVPxlNo % 2:
            self.padFOVPxlNo += 1

        # Init line of sight - Get the phase at the right size for the FOV
        self.los = lineofsight.LineOfSight(
                self.config, self.soapyConfig,
                propagationDirection="down")

        # Make a default circular mask of the pupil size if not provided
        if mask is None:
            mask = circle.circle(
                    self.simConfig.pupilSize/2., self.simConfig.pupilSize)
        else:
            # If the provided mask is the simSize, must crop down to pupilSize
            if mask.shape == (self.simConfig.simSize,)*2:
                mask = self.mask[
                        self.simConfig.simPad:-self.simConfig.simPad,
                        self.simConfig.simPad:-self.simConfig.simPad
                        ]

        self.scaledMask = numpy.round(interp.zoom(mask, self.FOVPxlNo)
                                      ).astype("int32")

        # Init FFT object
        self.FFTPadding = self.config.pxls * self.config.fftOversamp
        if self.FFTPadding < self.FOVPxlNo:
            while self.FFTPadding < self.FOVPxlNo:
                self.config.fftOversamp += 1
                self.FFTPadding\
                    = self.config.pxls * self.config.fftOversamp
            logger.info(
                "SCI FFT Padding less than FOV size... Setting oversampling to %d" % self.config.fftOversamp)

        self.FFT = AOFFT.FFT(
                inputSize=(self.FFTPadding, self.FFTPadding), axes=(0, 1),
                mode="pyfftw", dtype="complex64",
                fftw_FLAGS=(self.config.fftwFlag, "FFTW_DESTROY_INPUT"),
                THREADS=self.config.fftwThreads)

        # Convert phase in nm to radians at science wavelength
        self.phsNm2Rad = 2*numpy.pi/(self.sciConfig.wavelength*10**9)

        # Calculate ideal PSF for purposes of strehl calculation
        self.los.EField[:] = numpy.ones(
                (self.los.nOutPxls,) * 2, dtype=CDTYPE)
        self.calcFocalPlane()
        self.bestPSF = self.focalPlane.copy()
        self.psfMax = self.bestPSF.max()
        self.longExpStrehl = 0
        self.instStrehl = 0

    def calcTiltCorrect(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred
        on one pixel only
        """
        # Only required if pxl number is even
        if not self.config.pxls % 2:
            # Need to correct for half a pixel angle
            theta = float(self.FOVrad) / (2 * self.FFTPadding)

            # Find magnitude of tilt from this angle
            A = theta * self.telConfig.telDiam / \
                (2 * self.config.wavelength) * 2 * numpy.pi

            coords = numpy.linspace(-1, 1, self.FOVPxlNo)
            X, Y = numpy.meshgrid(coords, coords)
            self.tiltFix = -1 * A * (X + Y)
        else:
            self.tiltFix = numpy.zeros((self.FOVPxlNo,) * 2)

    def calcFocalPlane(self):
        '''
        Takes the calculated pupil phase, scales for the correct FOV,
        and uses an FFT to transform to the focal plane.
        '''
        self.EField = self.los.EField

        # Apply the system mask
        self.EField *= self.mask

        # Scaled the padded phase to the right size for the requried FOV
        self.EField_fov = interp.zoom(self.EField, self.padFOVPxlNo)

        # Chop out the phase across the pupil before the fft
        coord = int(round((self.padFOVPxlNo - self.FOVPxlNo) / 2.))
        self.EField_fov = self.EField_fov[coord:-coord, coord:-coord]

        # Get the focal plan using an FFT
        self.FFT.inputData[:self.FOVPxlNo, :self.FOVPxlNo] = self.EField_fov
        focalPlane_efield = AOFFT.ftShift2d(self.FFT())

        # Bin down to the required number of pixels
        self.focalPlane_efield = interp.binImgs(
            focalPlane_efield, self.config.fftOversamp)

        self.focalPlane = numpy.abs(self.focalPlane_efield.copy())**2

        # Normalise the psf
        self.focalPlane /= self.focalPlane.sum()

    def calcInstStrehl(self):
        """
        Calculates the instantaneous Strehl, including TT if configured.
        """
        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.focalPlane[self.sciConfig.pxls//2,self.sciConfig.pxls//2]/self.focalPlane.sum()/self.psfMax
        else:
            self.instStrehl = self.focalPlane.max()/self.focalPlane.sum()/ self.psfMax

    def frame(self, scrns, correction=None):
        """
        Runs a single science camera frame with one or more phase screens

        Parameters:
            scrns (ndarray, list, dict): One or more 2-d phase screens. Phase in units of nm.
            phaseCorrection (ndarray): Correction phase in nm

        Returns:
            ndarray: Resulting science PSF
        """
        self.los.frame(scrns, correction=correction)

        self.calcFocalPlane()

        self.calcInstStrehl()

        # Here so when viewing data, that outside of the pupil isn't visible.
        # self.residual*=self.mask

        return self.focalPlane


class singleModeFibre(PSF):

    def __init__(self, soapyConfig, nSci=0, mask=None):
        scienceCam.__init__(self, soapyConfig, nSci, mask)

        self.normMask = self.mask / numpy.sqrt(numpy.sum(numpy.abs(self.mask)**2))
        self.fibreSize = opt.minimize_scalar(self.refCouplingLoss, bracket=[1.0, self.simConfig.simSize]).x
        self.refStrehl = 1.0 - self.refCouplingLoss(self.fibreSize)
        self.fibre_efield = self.fibreEfield(self.fibreSize)
        print("Coupling efficiency: {0:.3f}".format(self.refStrehl))

    def fibreEfield(self, size):
        fibre_efield = circle.gaussian2d((self.simConfig.simSize, self.simConfig.simSize), (size, size))
        fibre_efield /= numpy.sqrt(numpy.sum(numpy.abs(circle.gaussian2d((self.simConfig.simSize*3, self.simConfig.simSize*3), (size, size)))**2))
        return fibre_efield

    def refCouplingLoss(self, size):
        return 1.0 - numpy.abs(numpy.sum(self.fibreEfield(size) * self.normMask))**2

    def calcInstStrehl(self):
        self.instStrehl = numpy.abs(numpy.sum(self.fibre_efield * numpy.exp(1j*self.residual*self.phsNm2Rad) * self.normMask))**2


# Compatability with older versions
scienceCam = ScienceCam = PSF


class PSF2(object):
    def __init__(self, soapy_config, sci_index, mask=None, los=None):

        self.los = los

        # Get parameters from configuration
        # ---------------------------------
        self.sci_config = soapy_config.scis[sci_index]

        self.pupil_size = soapy_config.sim.pupilSize
        self.threads = soapy_config.sim.threads

        self.telescope_diameter = soapy_config.tel.telDiam
        self.pxl_scale = self.sci_config.pxlScale
        self.wavelength = self.sci_config.wavelength
        self.nx_detector_pixels = self.sci_config.pxls

        # Calculate some parameters
        # The required size of the phase to get the correct FOV out of the FFT
        self.nx_interp_phase = int(round(
                self.pxl_scale * self.pupil_size * (numpy.pi/(180*3600)) * self.telescope_diameter
                / self.wavelength))

        # Find first multiple of detector pixels bigger than nx_subap_interp
        self.detector_bin_ratio = 0
        self.nx_focus_efield = 1
        while self.nx_focus_efield < self.nx_interp_phase:
            self.detector_bin_ratio += 1
            self.nx_focus_efield = self.nx_detector_pixels * self.detector_bin_ratio
        print("Set detector bin ratio to {}, nx_subap_focus_efield: {}".format(
            self.detector_bin_ratio, self.nx_focus_efield))

        # Find rad to nm conversion
        self.nm_to_rad = 1e-9 * (2 * numpy.pi) / self.wavelength

        # Allocate some data arrays
        self.focus = numpy.zeros((self.nx_focus_efield, self.nx_focus_efield), dtype="float32")
        self.detector = numpy.zeros((self.nx_detector_pixels, self.nx_detector_pixels))
        self.interp_phase = numpy.zeros((self.nx_interp_phase, self.nx_interp_phase), dtype="float32")

        # Initialise the FFT
        self.interp_efield = pyfftw.zeros_aligned(
                (self.nx_focus_efield, self.nx_focus_efield), dtype="complex64")
        self.focus_efield = self.interp_efield.copy()
        self.fft = pyfftw.FFTW(
                self.interp_efield, self.focus_efield, threads=self.threads,
                axes=(0, 1))

        self.strehl_reference = 1
        self.strehl_reference = self.frame(
                numpy.ones((self.pupil_size, self.pupil_size))).max()

    def interp_to_size(self):

        # Convert to rad
        self.phase *= self.nm_to_rad
        # print("Interp_to_size")
        numbalib.zoom(self.phase, self.interp_phase, threads=self.threads)


    def calculate_focal_plane(self):

        self.interp_efield[:self.nx_interp_phase, :self.nx_interp_phase] = numpy.exp(1j * self.interp_phase)
        self.fft()
        # print("Abs squared")
        numbalib.abs_squared(self.focus_efield, self.focus)
        self.focus = numpy.fft.fftshift(self.focus)

        # print("Bin")
        numbalib.bin_img(self.focus, self.detector_bin_ratio, self.detector)

    def calculate_strehl(self):
        self.instStrehl = self.detector.max() / self.strehl_reference

    def frame(self, phase):

        self.phase = phase

        self.interp_to_size()
        self.calculate_focal_plane()

        self.calculate_strehl()

        return self.detector