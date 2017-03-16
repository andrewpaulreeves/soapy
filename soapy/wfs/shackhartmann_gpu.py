"""
The Shack Hartmann WFS accelerated using numba cuda
"""

import numpy
from numba import cuda
from accelerate.cuda.fft.binding import Plan, CUFFT_C2C
from .. import AOFFT, LGS, logger, lineofsight_fast
from . import shackhartmannfast
from .. import gpulib, numbalib

# The data type of data arrays (complex and real respectively)
CDTYPE = numpy.complex64
DTYPE = numpy.float32

class ShackHartmannGPU(shackhartmannfast.ShackHartmannFast):
    def initLos(self):
        """
        Initialises the ``LineOfSight`` object, which gets the phase or EField in a given direction through turbulence.
        """
        self.los = lineofsight_fast.LineOfSightGPU(
                self.config, self.soapy_config,
                propagation_direction="down")



    def allocDataArrays(self):
        super(ShackHartmannGPU, self).allocDataArrays()


        self.interp_phase_gpu = cuda.to_device(self.interp_phase)
        self.interp_efield_gpu = cuda.to_device(self.interp_phase.astype("complex64"))
        self.subap_interp_efield_gpu = cuda.to_device(self.subap_interp_efield)
        self.binnedFPSubapArrays_gpu = cuda.to_device(self.binnedFPSubapArrays)
        self.subap_focus_intensity_gpu = cuda.to_device(self.subap_focus_intensity)
        self.temp_subap_intensity_gpu = cuda.to_device(self.temp_subap_intensity)
        self.detector_gpu = cuda.to_device(self.detector)
        self.centSubapArrays_gpu = cuda.to_device(self.centSubapArrays)
        self.slopes_gpu = cuda.to_device(self.slopes)


    # def calcFocalPlane(self, intensity=1):
    #
    #     if self.config.propagationMode == "Geometric":
    #         # Have to make phase the correct size if geometric prop
    #         gpulib.wfs.zoom_to_efield(self.los.phase_gpu, self.interp_efield_gpu)
    #         scaledEField = self.interp_efield_gpu.copy_to_host()
    #
    #     else:
    #         scaledEField = self.EField_gpu.copy_to_host()
    #
    #     # create an array of individual subap EFields
    #     self.FFT.inputData[:] = 0
    #     numbalib.wfs.chop_subaps_mask_pool(
    #             scaledEField, self.interp_subap_coords, self.nx_subap_interp,
    #             self.FFT.inputData, self.scaledMask, thread_pool=self.thread_pool)
    #     self.FFT.inputData[:, :self.nx_subap_interp, :self.nx_subap_interp] *= self.tilt_fix_efield
    #     self.FFT()
    #
    #     self.temp_subap_focus = AOFFT.ftShift2d(self.FFT.outputData)
    #
    #
    #     self.subap_focus_intensity[:] = abs(self.temp_subap_focus)**2
    #
    #     if intensity != 1:
    #         self.subap_focus_intensity *= intensity