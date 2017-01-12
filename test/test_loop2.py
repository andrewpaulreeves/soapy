import os

from soapy import loop

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../conf/")

CONFIG = 'vlt'

def test_init():

    sim = loop.Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()
def test_loop_frame():

    sim = loop.Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()

    sim.loop_frame(0)

def test_loop():

    sim = loop.Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()

    sim.loop()

if __name__ == "__main__":
    sim = loop.Simulation("{}/soapy2_{}.yaml".format(CONFIG_PATH, CONFIG))
    sim.aoinit()
    print("Start Loop")
    sim.loop()
