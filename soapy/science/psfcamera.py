"""
PSF Camera
----------

A simple camera that observes a point source, and therefore shows the current PSF 
results from an input phase or Electric Field.
"""


class PSF:
    def __init__(self, soapyConfig, nSci=0, mask=None):
        self.soapy_config = soapyConfig
        self.config = self.soapy_config.scis[nSci]
        self.sim_config = soapyConfig.sim_config

        # 1D size of input phase/efield
        self.nx_elements = self.soapy_config.sim.pupilSize

        # 1D size of detector
        self.nx_pixels = self.config.pxls

        self.pxl_scale = self.config.pixel_scale

