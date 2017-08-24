import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread

# python3 has queue, python2 has Queue
try:
    import queue
except ImportError:
    import Queue as queue

import numpy
import numba

@numba.jit(nopython=True, nogil=True, parallel=True)
def bilinear_interp(data, xCoords, yCoords, interpArray):
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
    for i in numba.prange(xCoords.shape[0]):
        x = xCoords[i]
        if x >= data.shape[0] - 1:
            x = data.shape[0] - 1 - 1e-9
        x1 = numba.int32(x)
        for j in range(yCoords.shape[0]):
            y = yCoords[j]
            if y >= data.shape[1] - 1:
                y = data.shape[1] - 1 - 1e-9
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


@numba.jit(nopython=True, nogil=True, parallel=True)
def bilinear_interp_inbounds(data, xCoords, yCoords, interpArray):
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
    for i in numba.prange(xCoords.shape[0]):
        x = xCoords[i]
        x1 = numba.int32(x)
        for j in range(yCoords.shape[0]):
            y = yCoords[j]
            y1 = numba.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray


@numba.jit(nopython=True, nogil=True, parallel=True)
def rotate(data, interpArray, rotation_angle):
    for i in numba.prange(interpArray.shape[0]):
        for j in range(interpArray.shape[1]):

            i1 = i - (interpArray.shape[0] / 2. - 0.5)
            j1 = j - (interpArray.shape[1] / 2. - 0.5)
            x = i1 * numpy.cos(rotation_angle) - j1 * numpy.sin(rotation_angle)
            y = i1 * numpy.sin(rotation_angle) + j1 * numpy.cos(rotation_angle)

            x += data.shape[0] / 2. - 0.5
            y += data.shape[1] / 2. - 0.5

            if x >= data.shape[0] - 1:
                x = data.shape[0] - 1.1
            x1 = numpy.int32(x)

            if y >= data.shape[1] - 1:
                y = data.shape[1] - 1.1
            y1 = numpy.int32(y)

            xGrad1 = data[x1 + 1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1 * (x - x1)

            xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
            a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

            yGrad = a2 - a1
            interpArray[i, j] = a1 + yGrad * (y - y1)
    return interpArray

@numba.vectorize(["float32(complex64)"], nopython=True, target="parallel")
def abs_squared(data):
    return abs(data)**2



@numba.jit(nopython=True, nogil=True, parallel=True)
def bin_img(imgs, bin_size, new_img):

    # loop over each element in new array
    for i in numba.prange(new_img.shape[0]):
        x1 = i * bin_size
        for j in range(new_img.shape[1]):
            y1 = j * bin_size
            new_img[i, j] = 0

            # loop over the values to sum
            for x in range(bin_size):
                for y in range(bin_size):
                    new_img[i, j] += imgs[x1 + x, y1 + y]



@numba.jit(nopython=True, nogil=True, parallel=True)
def fft_shift(data):
    s1 = data.shape[0]//2
    s2 = data.shape[1]//2
    for x in numba.prange(s1):
        for y in range(s2):

            temp_val1 = data[x, y]
            temp_val2 = data[(x + s1) % data.shape[0], (y + s2) % data.shape[1]]
            temp_val3 = data[x, (y + s2) % data.shape[1]]
            temp_val4 = data[(x + s1) % data.shape[0], y]

            data[x, y] = temp_val2
            data[(x + s1) % data.shape[0], (y + s2) % data.shape[1]] = temp_val1
            data[x, (y + s2) % data.shape[1]] = temp_val4
            data[(x + s1) % data.shape[0], y] = temp_val3

    return data


@numba.jit(nopython=True, nogil=True, parallel=True)
def fft_shift_1thread(data):
    s1 = data.shape[0]//2
    s2 = data.shape[1]//2
    for x in range(s1):
        for y in range(s2):

            temp_val1 = data[x, y]
            temp_val2 = data[(x + s1) % data.shape[0], (y + s2) % data.shape[1]]
            temp_val3 = data[x, (y + s2) % data.shape[1]]
            temp_val4 = data[(x + s1) % data.shape[0], y]

            data[x, y] = temp_val2
            data[(x + s1) % data.shape[0], (y + s2) % data.shape[1]] = temp_val1
            data[x, (y + s2) % data.shape[1]] = temp_val4
            data[(x + s1) % data.shape[0], y] = temp_val3

    return data

