from numba import cuda
import numba
import numpy
import math

import accelerate.cuda.blas

# Cuda threads per block
CUDA_TPB = 32

def zero_data(data, threads_per_block):

    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    bpg = int(numpy.ceil(data.size / threads_per_block))

    zero_data_kernel[threads_per_block, bpg](data)

@cuda.jit
def zero_data_kernel(data):
    tx = cuda.threadsIdx.x
    data[tx] = 0

def bilinterp2d_regular(
        data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray, threads_per_block=None):
    """
    2-D interpolation on a regular grid using purely python - 
    fast if compiled with numba
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
    bilinterp2d_regular_kernel[tpb, bpg](data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray)

    return interpArray

@cuda.jit
def bilinterp2d_regular_kernel(
        data, xMin, xMax, xSize, yMin, yMax, ySize, interpArray):
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
    # Thread id in a 2D grid
    i, j = cuda.grid(2)

    x = xMin + i*float(xMax - xMin)/(xSize - 1)
    x1 = numba.int32(x)

    y = yMin + j*float(yMax - yMin)/(ySize - 1)
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    interpArray[i,j] += a1 + yGrad*(y-y1)


def zoom(data, zoomArray, threads_per_block=None):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    Returns:
        ndarray: A pointer to the zoomArray
    """
    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    tpb = (threads_per_block,) * 2
    # blocks per grid
    bpg = (
            numpy.ceil(float(zoomArray.shape[0])/tpb),
            numpy.ceil(float(zoomArray.shape[1])/tpb)
            )

    zoom_kernel[tpb, bpg](data, zoomArray)

    return zoomArray

@cuda.jit
def zoom_kernel(data, zoomArray):
    """
    2-D zoom interpolation using purely python - fast if compiled with numba.
    Both the array to zoom and the output array are required as arguments, the
    zoom level is calculated from the size of the new array.
    Parameters:
        array (ndarray): The 2-D array to zoom
        zoomArray (ndarray): The array to place the calculation
    """
    i, j = cuda.grid(2)

    x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
    x1 = numba.int32(x)

    y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
    y1 = numba.int32(y)

    xGrad1 = data[x1+1, y1] - data[x1, y1]
    a1 = data[x1, y1] + xGrad1*(x-x1)

    xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
    a2 = data[x1, y1+1] + xGrad2*(x-x1)

    yGrad = a2 - a1
    zoomArray[i,j] = a1 + yGrad*(y-y1)


def phs2EField(phase, EField):
    """
    Converts phase to an efield on the GPU
    """
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            numpy.ceil(float(phase.shape[0])/tpb),
            numpy.ceil(float(phase.shape[1])/tpb)
            )

    phs2EField_kernel[tpb, bpg](phase, EField)

    return EField

@cuda.jit
def phs2EField_kernel(phase, EField):
    i, j = cuda.grid(2)

    EField[i, j] = math.exp(phs[i, j])

def absSquared3d(inputData, outputData, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock,)*3
    # blocks per grid
    bpg = (
            int(numpy.ceil(float(inputData.shape[0])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[1])/threadsPerBlock)),
            int(numpy.ceil(float(inputData.shape[2])/threadsPerBlock))
            )

    absSquared3d_kernel[tpb, bpg](inputData, outputData)

    return outputData

@cuda.jit
def absSquared3d_kernel(inputData, outputData):
    i, j, k = cuda.grid(3)
    outputData[i, j, k] = inputData[i, j, k].real**2 + inputData[i, j, k].imag**2


def array_sum(array1, array2, output_data=None,  threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = threadsPerBlock
    # blocks per grid
    bpg = int(numpy.ceil(float(array1.shape[0])/threadsPerBlock)),

    if output_data is None: # Assume in place
        array_sum_inplace_kernel[tpb, bpg](array1, array2)
        return array1
    else:
        array_sum_kernel[tpb, bpg](array1, array2, output_data)
        return output_data

@cuda.jit
def array_sum_inplace_kernel(array1, array2):
    i = cuda.grid(1)

    array1[i] += array2[i]

@cuda.jit
def array_sum_kernel(array1, array2, output_array):
    i = cuda.grid(1)
    output_array[i] = array1[i] + array2[i]

def array_sum2d(array1, array2, threadsPerBlock=None):

    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock, )*2
    # blocks per grid
    bpg = (
            int(numpy.ceil(float(array1.shape[0])/threadsPerBlock)),
            int(numpy.ceil(float(array1.shape[1])/threadsPerBlock))
            )

    array_sum2d_kernel[tpb, bpg](array1, array2)

    return array1

@cuda.jit
def array_sum2d_kernel(array1, array2):
    i, j = cuda.grid(2)

    if i < array1.shape[0]:
        if j < array1.shape[1]:
            array1[i, j] += array2[i, j]

def mvm(matrix, vector, output_vector, blas=None):
    """
    Wrapper aoudn the cublas gemv call for matrix-vector multiplication
    
    Parameters:
        matrix (devicearray): 2-d cuda array in FORTRAN ordering 
        vector (devicearray): 1-d cuda array 
        output_vector (devicearray): array to place calculation  
        blas (accelerate.cuda.blas.Blas, optional): Blas calculation object. If not given will create one.

    Returns:
        devicearray: the calculated device array, output_vector

    """
    if blas is None:
        blas = accelerate.cuda.blas.Blas()

    m = matrix.shape[0]
    n = matrix.shape[1]
    alpha = 1 # Factor to multiply the MVM with
    beta = 0  # Factor to multiply existing and sum with existing contents of output_vector

    blas.gemv("N", m, n, alpha, matrix, vector, beta, output_vector)

    return output_vector



def rotate(data, output_data, rotation_angle,threadsPerBlock=None):
    if threadsPerBlock is None:
        threadsPerBlock = CUDA_TPB

    tpb = (threadsPerBlock,) * 2
    # blocks per grid
    bpg = (
        int(numpy.ceil(float(output_data.shape[0]) / threadsPerBlock)),
        int(numpy.ceil(float(output_data.shape[1]) / threadsPerBlock))
    )

    rotate[tpb, bpg](data, output_data, rotation_angle)

    return output_data


@cuda.jit
def rotate(data, interpArray, rotation_angle):
    i, j = cuda.grid(2)

    if i < interpArray.shape[0] and j < interpArray.shape[1]:

        i1 = i - (interpArray.shape[0] / 2. - 0.5)
        j1 = j - (interpArray.shape[1] / 2. - 0.5)
        x = i1 * math.cos(rotation_angle) - j1 * math.sin(rotation_angle)
        y = i1 * math.sin(rotation_angle) + j1 * math.cos(rotation_angle)

        x += data.shape[0] / 2. - 0.5
        y += data.shape[1] / 2. - 0.5

        if x >= data.shape[0] - 1:
            x = data.shape[0] - 1.1
        x1 = numba.int32(x)

        if y >= data.shape[1] - 1:
            y = data.shape[1] - 1.1
        y1 = numba.int32(y)

        xGrad1 = data[x1 + 1, y1] - data[x1, y1]
        a1 = data[x1, y1] + xGrad1 * (x - x1)

        xGrad2 = data[x1 + 1, y1 + 1] - data[x1, y1 + 1]
        a2 = data[x1, y1 + 1] + xGrad2 * (x - x1)

        yGrad = a2 - a1
        interpArray[i, j] = a1 + yGrad * (y - y1)
