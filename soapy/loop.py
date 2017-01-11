import numpy

from . import simulation, sh_wfs, lineofsight2

class Simulation(simulation.Sim):

    def aoinit(self):
        super(Simulation, self).aoinit()

        wfs_config = sh_wfs.WFS_Config()

        wfs_config.pupil_size = self.config.sim.pupilSize

        wfs_config.telescope_diameter = self.config.tel.telDiam
        wfs_config.nx_subaps = self.config.wfss[0].nxSubaps
        wfs_config.subap_diam = wfs_config.telescope_diameter / wfs_config.nx_subaps
        wfs_config.pxl_scale = self.config.wfss[0].subapFOV/self.config.wfss[0].pxlsPerSubap
        wfs_config.nx_subap_pxls = self.config.wfss[0].pxlsPerSubap
        wfs_config.wavelength = self.config.wfss[0].wavelength
        wfs_config.subap_threshold = self.config.wfss[0].subapThreshold
        wfs_config.mask = self.mask

        wfs_config.threads = 1

        self.sh = sh_wfs.ShackHartmann(wfs_config)

        self.sh_los = lineofsight2.LineOfSight(wfs_config)

        self.slopes = numpy.zeros((self.config.sim.nIters, len(self.slopes)))

    def loop(self):

        for i in range(self.config.sim.nIters):

            self.loop_frame(i)


    def loop_frame(self, i):

        phase_screens = self.atmos.moveScrns()

        wfs_phs = self.sh_los(phase_screens)
        slopes = self.sh(wfs_phs)

        self.slopes[i] = slopes