"""
Module containing accelerated version of AOTools functions using "numba" for Soapy
"""
from threading import Thread

import numba
import numpy

def zoom(data, zoomArray, threads=None):
    """
    A function which deals with threaded numba interpolation.

    Parameters:
        array (ndarray): The 2-D array to interpolate
        zoomArray (ndarray, tuple): The array to place the calculation, or the shape to return
        threads (int): Number of threads to use for calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if isinstance(zoomArray, numpy.ndarray) is not True:
        zoomArray = numpy.zeros((zoomArray))

    if threads!=1 and threads!=None:

        nx = zoomArray.shape[0]

        Ts = []
        for t in range(threads):
            Ts.append(Thread(target=zoom_numbaThread,
                args = (
                    data,
                    numpy.array([int(t*nx/threads), int((t+1)*nx/threads)]),
                    zoomArray)
                ))
            Ts[t].start()

        for T in Ts:
            T.join()

    else:
        zoom_numba1(data, zoomArray)

    return zoomArray

@numba.jit(nopython=True, nogil=True)
def zoom_numbaThread(data,  chunkIndices, zoomArray):
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

    for i in xrange(chunkIndices[0], chunkIndices[1]):
        x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
        x1 = numba.int32(x)
        for j in xrange(zoomArray.shape[1]):
            y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            zoomArray[i,j] = a1 + yGrad*(y-y1)


    return zoomArray

@numba.jit(nopython=True)
def zoom_numba1(data, zoomArray):
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

    for i in xrange(numba.int32(zoomArray.shape[0])):
        x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
        x1 = numba.int32(x)
        for j in xrange(zoomArray.shape[1]):
            y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            zoomArray[i,j] = a1 + yGrad*(y-y1)


    return zoomArray


def linterp2d(data, xCoords, yCoords, interpArray, threads=None):
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

    if threads!=1 and threads!=None:

        nx = xCoords.shape[0]

        Ts = []
        for t in range(threads):
            Ts.append(Thread(target=linterp2d_thread,
                args = (
                    data, xCoords, yCoords,
                    numpy.array([int(t*nx/threads), int((t+1)*nx/threads)]),
                    interpArray)
                ))
            Ts[t].start()

        for T in Ts:
            T.join()

    else:
        linterp2d_1thread(data, xCoords, yCoords, interpArray)

    return interpArray

@numba.jit(nopython=True, nogil=True)
def linterp2d_thread(data, xCoords, yCoords, chunkIndices, interpArray):
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


    if xCoords[-1] == data.shape[0]-1:
        xCoords[-1] -= 1e-6
    if yCoords[-1] == data.shape[1]-1:
        yCoords[-1] -= 1e-6

    jRange = xrange(yCoords.shape[0])
    for i in xrange(chunkIndices[0],chunkIndices[1]):
        x = xCoords[i]
        x1 = numba.int32(x)
        for j in jRange:
            y = yCoords[j]
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            interpArray[i,j] = a1 + yGrad*(y-y1)

    return interpArray


@numba.jit(nopython=True)
def linterp2d_1thread(data, xCoords, yCoords, interpArray):
    """
    2-D interpolation using purely python - fast if compiled with numba

    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 1-D array of x-coordinates
        yCoords (ndarray): A 2-D array of y-coordinates
        interpArray (ndarray): The array to place the calculation

    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """


    if xCoords[-1] == data.shape[0]-1:
        xCoords[-1] -= 1e-6
    if yCoords[-1] == data.shape[1]-1:
        yCoords[-1] -= 1e-6

    jRange = xrange(yCoords.shape[0])
    for i in xrange(xCoords.shape[0]):
        x = xCoords[i]
        x1 = numba.int32(x)
        for j in jRange:

            y = yCoords[j]
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            interpArray[i,j] = a1 + yGrad*(y-y1)
    return interpArray


@numba.vectorize([numba.float32(numba.complex64),
            numba.float64(numba.complex128)])
def absSquare(data):
        return (data.real**2 + data.imag**2)


@numba.jit(nopython=True)
def binImg(img, binSize, newImg):

    for i in xrange(newImg.shape[0]):
        x1 = i*binSize
        for j in xrange(newImg.shape[1]):
            y1 = j*binSize
            newImg[i,j] = 0
            for x in xrange(binSize):
                for y in xrange(binSize):
                    newImg[i,j] += img[x+x1, y+y1]