import numba
import numpy

@numba.jit(nopython=True, parallel=True)
def calculate_seperations(positions, seperations):
    for i in numba.prange(positions.shape[0]):
        (x1, y1) = positions[i]
        for j in range(positions.shape[0]):
            (x2, y2) = positions[j]
            delta_x = x2 - x1
            delta_y = y2 - y1

            delta_r = numpy.sqrt(delta_x ** 2 + delta_y ** 2)

            seperations[i, j] = delta_r