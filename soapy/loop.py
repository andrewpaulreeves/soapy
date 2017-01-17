import time

import numpy

from . import simulation, lineofsight2
from .wfs import ShackHartmann2

class Simulation(simulation.Sim):

    def aoinit(self):
        super(Simulation, self).aoinit()




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