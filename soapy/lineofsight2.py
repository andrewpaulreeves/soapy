from threading import Thread
import multiprocessing
N_CPU = multiprocessing.cpu_count()
import time

import numpy
import numba

ASEC2RAD = (numpy.pi/(180 * 3600))


class LineOfSight(object):

    def __init__(self, obj_config, soapy_config):

        self.direction = obj_config.position
        self.src_altitude = obj_config.GSHeight

        self.n_layers = soapy_config.atmos.scrnNo
        self.layer_altitudes = soapy_config.atmos.scrnHeights

        self.phase_pxl_scale = soapy_config.sim.pxlScale**(-1)
        self.pupil_size = soapy_config.sim.pupilSize
        self.nx_scrn_size = soapy_config.sim.scrnSize

        self.threads = soapy_config.sim.threads

        # Calculate coords of phase at each altitude
        self.layer_metapupil_coords = numpy.zeros((self.n_layers, 2, self.pupil_size))
        for i in range(self.n_layers):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.layer_altitudes[i])
            self.layer_metapupil_coords[i, 0] = numpy.linspace(x1, x2, self.pupil_size) + self.nx_scrn_size/2.
            self.layer_metapupil_coords[i, 1] = numpy.linspace(y1, y2, self.pupil_size) + self.nx_scrn_size/2.

        self.phase_screens = numpy.zeros((self.n_layers, self.pupil_size, self.pupil_size))

    def calculate_altitude_coords(self, layer_altitude):
        direction_radians = ASEC2RAD * numpy.array(self.direction)

        centre = (direction_radians * layer_altitude) / self.phase_pxl_scale

        if self.src_altitude != 0:
            meta_pupil_size = self.pupil_size * (1 - layer_altitude/self.src_altitude)
        else:
            meta_pupil_size = self.pupil_size

        x1 = centre[0] - meta_pupil_size/2.
        x2 = centre[0] + meta_pupil_size/2.
        y1 = centre[1] - meta_pupil_size/2.
        y2 = centre[1] + meta_pupil_size/2.

        return x1, x2, y1, y2

    def get_phase_slices(self):
        """
        Calculates the phase seen by the los at each altitude. Compiles a list of each phase.
        """
        get_phase_slices(
                self.raw_phase_screens, self.layer_metapupil_coords, self.phase_screens, self.threads)

    def propagate_light(self):
        self.output_phase = self.phase_screens.sum(0)

    def frame(self, phase_screens, phase_correction=None):
        self.raw_phase_screens = phase_screens

        self.get_phase_slices()
        self.propagate_light()

        return self.output_phase


# LOS Functions
# -------------
def get_phase_slices(raw_phase_screens, layer_metapupil_coords, phase_screens, threads=None):

    if threads is None:
        threads = N_CPU

    for i in range(raw_phase_screens.shape[0]):
        bilinear_interp(raw_phase_screens[i], layer_metapupil_coords[i, 0], layer_metapupil_coords[i, 1], phase_screens[i], threads)


def bilinear_interp(data, xCoords, yCoords, interpArray, threads=None):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation
        threads (int): Number of threads to use for calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    nx = xCoords.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bilinear_interp_numba,
                         args=(
                             data, xCoords, yCoords,
                             numpy.array([int(t * nx / threads), int((t + 1) * nx / threads)]),
                             interpArray)
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return interpArray

@numba.jit(nopython=True, nogil=True)
def bilinear_interp_numba(data, xCoords, yCoords, chunkIndices, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba
    This version also accepts a parameter specifying how much of the array
    to operate on. This is useful for multi-threading applications.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        chunkIndices (ndarray): A 2 element array, with (start Index, stop Index) to work on for the x-dimension.
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """

    if xCoords[-1] == data.shape[0] - 1:
        xCoords[-1] -= 1e-6
    if yCoords[-1] == data.shape[1] - 1:
        yCoords[-1] -= 1e-6

    jRange = range(yCoords.shape[0])
    for i in range(chunkIndices[0], chunkIndices[1]):
        x = xCoords[i]
        x1 = numba.int32(x)
        for j in jRange:
            y = yCoords[j]
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)

    return interpArray


class LOS_Config(object):
    pass


def loop_test(los, screens, N=1000):
    t1 = time.time()

    for i in range(N):
        los.frame(screens)
    t2 = time.time()

    f_time = (t2 - t1)/N
    print("Frame runs in {:.2f}ms".format(f_time*1000))
    print("Iters per second: {:2.2f}".format(1./f_time))


if __name__ == "__main__":
    los_config = LOS_Config()

    los_config.pupil_size = 640
    los_config.phase_pxl_scale = 38./los_config.pupil_size
    los_config.n_layers = 37
    los_config.layer_altitudes = numpy.linspace(0, 20, los_config.n_layers)
    los_config.direction = (10, 10)
    los_config.src_altitude = 0
    los_config.nx_scrn_size = 800

    los_config.threads = 4

    # some example phase screen

    phase_screens = numpy.ones((los_config.n_layers, los_config.nx_scrn_size, los_config.nx_scrn_size), dtype="float32")

    los = LineOfSight(los_config)
    phs_out = los.frame(phase_screens)

    loop_test(los, phase_screens, 100)