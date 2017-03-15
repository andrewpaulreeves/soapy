import numpy
import numba
from numba import cuda


CUDA_TPB = 32

def bilinear_interp(data, xCoords, yCoords, interpArray, threads_per_block=None):
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
    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    tpb = (threads_per_block,) * 2
    # blocks per grid
    bpg = (
            int(numpy.ceil(interpArray.shape[0] / threads_per_block)),
            int(numpy.ceil(interpArray.shape[1] / threads_per_block))
            )

    bilinear_interp_kernel[tpb, bpg](data, xCoords, yCoords, interpArray)

    return interpArray

@cuda.jit
def bilinear_interp_kernel(data, xCoords, yCoords, interpArray):
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
    # Thread id in a 1D block
    i, j = cuda.grid(2)
    if i < interpArray.shape[0] and j < interpArray.shape[1]:
        # Get corresponding coordinates in image
        x = xCoords[i]
        x1 = numba.int32(x)
        y = yCoords[j]
        y1 = numba.int32(y)

        # Do bilinear interpolation
        xGrad1 = data[x1+1, y1] - data[x1, y1]
        a1 = data[x1, y1] + xGrad1*(x-x1)

        xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
        a2 = data[x1, y1+1] + xGrad2*(x-x1)

        yGrad = a2 - a1
        interpArray[i,j] = a1 + yGrad*(y-y1)