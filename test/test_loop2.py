import os
import time

import numpy
from matplotlib  import pyplot

from soapy import simulation, lineofsight2
from soapy.wfs import ShackHartmann2

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

CONFIG = 'elt'

def test_init():

    sim = Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()
def test_loop_frame():

    sim = Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()

    sim.loop_frame(0)

def test_loop():

    sim = Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()

    sim.loop()


class Simulation(simulation.Sim):
    def aoinit(self):
        super(Simulation, self).aoinit()

        self.sh_los = lineofsight2.LineOfSight(self.config.wfss[0], self.config)

        self.sh = ShackHartmann2(self.config, 0, mask=self.mask, los=self.sh_los)

        self.slopes = numpy.zeros((self.config.sim.nIters, len(self.sh.slopes)))

    def loop(self):
        self.atmos_time = self.los_time = self.wfs_time = 0

        print("Start loop...")
        t1 = time.time()
        for i in range(self.config.sim.nIters):
            self.loop_frame(i)

        time_elapsed = time.time() - t1
        iters_per_sec = self.config.sim.nIters / time_elapsed

        print("Run time: {} seconds".format(time_elapsed))
        print("iters per second: {}".format(iters_per_sec))

        print("\n")
        print("Breakdown:")
        print("{:.2f}% Moving atmosphere".format(100 * self.atmos_time / time_elapsed))
        print("{:.2f}% Propagating light through atmosphere".format(100 * self.los_time / time_elapsed))
        print("{:.2f}% Simulating WFS".format(100 * self.wfs_time / time_elapsed))

        return time_elapsed

    def loop_frame(self, i):
        t_start = time.time()
        phase_screens = self.atmos.moveScrns()
        phase_screens = numpy.array(list(phase_screens.values()))
        t_atmos = time.time()
        self.atmos_time += (t_atmos - t_start)

        wfs_phs = self.sh_los.frame(phase_screens)
        t_los = time.time()
        self.los_time += (t_los - t_atmos)

        slopes = self.sh.frame(wfs_phs)
        t_wfs = time.time()
        self.wfs_time += (t_wfs - t_los)

        self.slopes[i] = slopes


def time_vs_threads():
    times = []
    for i in range(1, 11):
        sim = Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
        sim.config.sim.threads = i
        sim.aoinit()

        t = sim.loop()
        times.append(t)

    pyplot.plot(range(1, 11), times)
    pyplot.show()

if __name__ == "__main__":
    # sim = Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    # sim.aoinit()
    # print("Start Loop")
    # sim.loop()

    time_vs_threads()