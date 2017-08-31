import numpy
import numba

from . import numbalib

@numba.jit(nopython=True, nogil=True, parallel=True)
def fft_shift_subaps(data):
    for s in numba.prange(data.shape[0]):
        numbalib.fft_shift_1thread(data[s])
    return data


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
    """
    Splits the phase into sub-apertures

    A given phase array is split into a grid of "sub-apertures", each with size
    `nx_subap_size`. The location  of each sub-apertures intial vertex is given
    in the array `subap_coords`, a 2-d array of shape `(n_subaps, 2)`. The resulting
    sub-apertures are placed into the pre-allocated `subap_array`, of size
    `(n_subaps, nx_subap_size, nx_subap_size)`. During this process, a pupil
    mask is applied, given by `mask`.

    Parameters:
        phase (ndarray): Array of phase
        subap_coords (ndarray): Coordinates of each sub-apertures initila vertex
        nx_subap_size (int): 1-D Size of sub-apertures
        subap_array (ndarray): Array to place sub-apertures
        mask (ndarray): Pupil mask

    Returns:
        ndarray: View of subap_array
    """
    for i in numba.prange(subap_coords.shape[0]):
        x1 = numba.int32(subap_coords[i, 0])
        x2 = numba.int32(subap_coords[i, 0] + nx_subap_size)
        y1 = numba.int32(subap_coords[i, 1])
        y2 = numba.int32(subap_coords[i, 1] + nx_subap_size)

        for x in range(nx_subap_size):
            for y in range(nx_subap_size):
                subap_array[i, x, y] = phase[x1+x, y1+y] * mask[x1+x, y1+y]

    return subap_array



@numba.jit(nopython=True, nogil=True, parallel=True)
def chop_subaps(phase, subap_coords, nx_subap_size, subap_array):
    for i in numba.prange(numba.int32(subap_coords.shape[0])):
        x = int(subap_coords[i, 0])
        y = int(subap_coords[i, 1])

        subap_array[i, :nx_subap_size, :nx_subap_size] = phase[x:x + nx_subap_size, y:y + nx_subap_size]

    return subap_array


@numba.jit(nopython=True, nogil=True)
def place_subaps_on_detector_(subap_imgs, detector_img, detector_positions, subap_coords):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_imgs.shape[0]):
        x1, x2, y1, y2 = detector_positions[i]
        sx1 ,sx2, sy1, sy2 = subap_coords[i]
        detector_img[x1: x2, y1: y2] += subap_imgs[i, sx1: sx2, sy1: sy2]

    return detector_img


@numba.jit(nopython=True, nogil=True)
def place_subaps_on_detector(subap_imgs, detector_img, detector_positions, subap_coords):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_imgs.shape[0]):
        x1, x2, y1, y2 = detector_positions[i]
        sx1 ,sx2, sy1, sy2 = subap_coords[i]
        for x in range(x2 - x1):
            for y in range(y2 - y1):
                detector_img[x1 + x, y1 + y] += subap_imgs[i, sx1 + x, sy1 + y]

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


@numba.jit(nopython=True, parallel=True)
def centreOfGravity(subaps, centroids, threshold=0, ref=None):
    nx_subap_size = subaps.shape[1]
    ny_subap_size = subaps.shape[2]

    # Loop over each sub-ap
    # for s in numba.prange(subaps.shape[0]):
    for s in range(subaps.shape[0]):
        centroid_sum_x = 0
        centroid_sum_y = 0
        subap_sum = 0
        threshold_max = threshold * subaps[s].max()
        for x in range(nx_subap_size):
            for y in range(ny_subap_size):

                if subaps[s, x, y] > threshold_max:
                    centroid_sum_x += subaps[s, x, y] * x
                    centroid_sum_y += subaps[s, x, y] * y
                    subap_sum += subaps[s, x, y]

        centroids[s, 1] = (centroid_sum_x / subap_sum) + 0.5
        centroids[s, 0] = (centroid_sum_y / subap_sum) + 0.5

    return centroids.T

