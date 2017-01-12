import time

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
        wfs_config.mask = self.mask[self.config.sim.simPad:-self.config.sim.simPad, self.config.sim.simPad: -self.config.sim.simPad]

        wfs_config.phase_pxl_scale = wfs_config.telescope_diameter / wfs_config.pupil_size
        wfs_config.n_layers = self.config.atmos.scrnNo
        wfs_config.layer_altitudes = self.config.atmos.scrnHeights
        wfs_config.direction = self.config.wfss[0].GSPosition
        wfs_config.src_altitude = self.config.wfss[0].GSHeight
        wfs_config.nx_scrn_size = self.config.sim.scrnSize

        wfs_config.threads = 4

        self.sh = sh_wfs.ShackHartmann(wfs_config)

        self.sh_los = lineofsight2.LineOfSight(wfs_config)

        self.slopes = numpy.zeros((self.config.sim.nIters, len(self.sh.slopes)))

    def loop(self):

        self.atmos_time = self.los_time = self.wfs_time = 0

        t1 = time.time()
        for i in range(self.config.sim.nIters):

            self.loop_frame(i)

        time_elapsed = time.time() - t1
        iters_per_sec = self.config.sim.nIters/time_elapsed

        print("Run time: {} seconds".format(time_elapsed))
        print("iters per second: {}".format(iters_per_sec))

        print("\n")
        print("Breakdown:")
        print("{:.2f}% Moving atmosphere".format(100*self.atmos_time/time_elapsed))
        print("{:.2f}% Propagating light through atmosphere".format(100*self.los_time/time_elapsed))
        print("{:.2f}% Simulating WFS".format(100*self.wfs_time/time_elapsed))

    def loop_frame(self, i):

        t_start = time.time()
        phase_screens = self.atmos.moveScrns()
        phase_screens = numpy.array(list(phase_screens.values()))
        t_atmos = time.time()
        self.atmos_time += (t_atmos-t_start)

        wfs_phs = self.sh_los.frame(phase_screens)
        t_los = time.time()
        self.los_time += (t_los - t_atmos)

        slopes = self.sh.frame(wfs_phs)
        t_wfs = time.time()
        self.wfs_time += (t_wfs - t_los)

        self.slopes[i] = slopes