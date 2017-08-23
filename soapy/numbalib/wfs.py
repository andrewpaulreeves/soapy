import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread
from . import numbalib

import numpy
import numba

@numba.jit(nopython=True, nogil=True, parallel=True)
def zoom(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.

    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``zoomArray''
    """

    for i in numba.prange(zoomArray.shape[0]):
        x = i * numba.float32(data.shape[0] - 1) / (zoomArray.shape[0] - 0.99999999)
        x1 = numba.int32(x)
        for j in range(zoomArray.shape[1]):
            y = j * numba.float32(data.shape[1] - 1) / (zoomArray.shape[1] - 0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            zoomArray[i, j] = a1 + yGrad * (y - y1)

    return zoomArray


@numba.jit(nopython=True, nogil=True, parallel=True)
def zoomtoefield(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.

    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``zoomArray''
    """

    for i in numba.prange(zoomArray.shape[0]):
        x = i * numba.float32(data.shape[0] - 1) / (zoomArray.shape[0] - 0.99999999)
        x1 = numba.int32(x)
        for j in range(zoomArray.shape[1]):
            y = j * numba.float32(data.shape[1] - 1) / (zoomArray.shape[1] - 0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            phase_value = (a1 + yGrad * (y - y1))
            zoomArray[i, j] = numpy.exp(1j * phase_value)

    return zoomArray


@numba.jit(nopython=True, nogil=True, parallel=True)
def chop_subaps_mask(phase, subap_coords, nx_subap_size, subap_array, mask):
    for i in numba.prange(subap_coords.shape[0]):
        x1 = int(subap_coords[i, 0])
        x2 = int(subap_coords[i, 0] + nx_subap_size)
        y1 = int(subap_coords[i, 1])
        y2 = int(subap_coords[i, 1] + nx_subap_size)

        subap_array[i, :nx_subap_size, :nx_subap_size] = phase[x1: x2, y1: y2] * mask[x1: x2, y1: y2]

    return subap_array


@numba.jit(nopython=True, nogil=True, parallel=True)
def chop_subaps(phase, subap_coords, nx_subap_size, subap_array):
    for i in numba.prange(subap_coords.shape[0]):
        x = int(subap_coords[i, 0])
        y = int(subap_coords[i, 1])

        subap_array[i, :nx_subap_size, :nx_subap_size] = phase[x:x + nx_subap_size, y:y + nx_subap_size]

    return subap_array


@numba.jit(nopython=True, nogil=True)
def place_subaps_on_detector(subap_imgs, detector_img, detector_positions, subap_coords):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_imgs.shape[0]):
        x1, x2, y1, y2 = detector_positions[i]
        sx1 ,sx2, sy1, sy2 = subap_coords[i]
        detector_img[x1: x2, y1: y2] += subap_imgs[i, sx1: sx2, sy1: sy2]

    return detector_img


@numba.jit(nopython=True, nogil=True, parallel=True)
def bin_imgs(imgs, bin_size, new_img):
    # loop over subaps
    for n in numba.prange(imgs.shape[0]):
        # loop over each element in new array
        for i in range(new_img.shape[1]):
            x1 = i * bin_size
            for j in range(new_img.shape[2]):
                y1 = j * bin_size
                new_img[n, i, j] = 0
                # loop over the values to sum
                for x in range(bin_size):
                    for y in range(bin_size):
                        new_img[n, i, j] += imgs[n, x + x1, y + y1]



class Centroider(object):
    def __init__(self, n_subaps, nx_subap_pxls, threads=None):

        if threads is None:
            self.threads = 1
        else:
            self.threads = threads

        self.n_subaps = n_subaps
        self.nx_subap_pxls = nx_subap_pxls

        self.indices = numpy.indices((self.nx_subap_pxls, self.nx_subap_pxls))

        self.centroids = numpy.zeros((n_subaps, 2))

    def __call__(self, subaps):
        self.centre_of_gravity_numpy(subaps)
        return self.centroids

    def centre_of_gravity_numpy(self, subaps):
        self.centroids[:, 1] = ((self.indices[0] * subaps).sum((1, 2)) / subaps.sum((1, 2))) + 0.5 - subaps.shape[
                                                                                                         1] * 0.5
        self.centroids[:, 0] = ((self.indices[1] * subaps).sum((1, 2)) / subaps.sum((1, 2))) + 0.5 - subaps.shape[
                                                                                                         2] * 0.5
        return self.centroids

    def centre_of_gravity_numba(self, subaps):

        centre_of_gravity(subaps, self.indices, self.centroids)
        return self.centroids


@numba.jit(nopython=True, nogil=True, parallel=True)
def centre_of_gravity(subaps, indices, centroids):
    nx_subap_size = subaps.shape[1]

    centroids[:, 0] = (
            indices[0] * subaps).sum((1, 2)) / subaps.sum((1, 2)) + 0.5 - nx_subap_size * 0.5
    centroids[:, 1] = (
            indices[1] * subaps).sum((1, 2)) / subaps.sum((1, 2)) + 0.5 - nx_subap_size * 0.5

