from soapy import confParse, WFS
from soapy.aotools import circle
import unittest
import numpy
import os
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

class TestWfs(unittest.TestCase):

    def test_SHWFSInit(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = aotools.circle(config.sim.pupilSize / 2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

    def testd_SHWfsFrame(self):
        config = confParse.Configurator(os.path.join(CONFIG_PATH, "sh_8x8.py"))
        config.loadSimParams()
        mask = circle.circle(config.sim.pupilSize/2., config.sim.simSize)

        wfs = WFS.ShackHartmann(config, mask=mask)

        wfs.frame(numpy.zeros((config.sim.simSize, config.sim.simSize)))



if __name__=="__main__":
    unittest.main()
