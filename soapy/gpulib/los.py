import numpy
import numba
from numba import cuda


CUDA_TPB = 32

def get_phase(raw_screens, coords, output_phase_screens, threads_per_block=None):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 2-D array of x-coordinates, y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    tpb = (threads_per_block,) * 3
    # blocks per grid
    bpg = (
            int(numpy.ceil(output_phase_screens.shape[0] / threads_per_block)),
            int(numpy.ceil(output_phase_screens.shape[1] / threads_per_block)),
            int(numpy.ceil(output_phase_screens.shape[2] / threads_per_block))
            )

    get_phase_kernel[tpb, bpg](raw_screens, coords, output_phase_screens)

    return output_phase_screens

@cuda.jit
def get_phase_kernel(raw_screens, coords, output_phase_screens):
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
    layer, i, j = cuda.grid(3)
    if (layer<output_phase_screens.shape[0]
            and i < output_phase_screens.shape[1]
            and j < output_phase_screens.shape[2]):
        # Get corresponding coordinates in image
        x = coords[layer, 0, i]
        if x >= raw_screens.shape[1] - 1:
            x = raw_screens.shape[1] - 1 - 1e-10
        x1 = numba.int32(x)
        y = coords[layer, 1, j]
        if y >= raw_screens.shape[2] - 1:
            y = raw_screens.shape[2] - 1 - 1e-10
        y1 = numba.int32(y)

        # Do bilinear interpolation
        xGrad1 = raw_screens[layer, x1+1, y1] - raw_screens[layer, x1, y1]
        a1 = raw_screens[layer, x1, y1] + xGrad1*(x-x1)

        xGrad2 = raw_screens[layer, x1+1, y1+1] - raw_screens[layer, x1, y1+1]
        a2 = raw_screens[layer, x1, y1+1] + xGrad2*(x-x1)

        yGrad = a2 - a1
        output_phase_screens[layer, i,j] = a1 + yGrad*(y-y1)


def geometric_propagation(
            phase_screens, output_phase, phs2rad, direction_multiplier,
            threads_per_block=None):

    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    tpb = (threads_per_block,) * 2
    # blocks per grid
    bpg = (
            int(numpy.ceil(output_phase.shape[0] / threads_per_block)),
            int(numpy.ceil(output_phase.shape[1] / threads_per_block)),
            )

    geometric_propagation_kernel[tpb, bpg](phase_screens, output_phase, phs2rad, direction_multiplier)

    return output_phase

@cuda.jit
def geometric_propagation_kernel(phase_screens, output_phase, phs2rad, direction_multiplier):
    i, j  = cuda.grid(2)

    if i < output_phase.shape[0] and j < output_phase.shape[1]:
        output_phase[i, j] = 0
        for layer in range(phase_screens.shape[0]):

            # print(layer, i, j)
            output_phase[i, j] += phase_screens[layer, i, j]

        output_phase[i, j] = output_phase[i, j] * phs2rad * direction_multiplier


def apply_correction(dm_correction, coords, output_phase, phs2rad, threads_per_block=None):
    """
    2-D interpolation using purely python - fast if compiled with numba
    Parameters:
        array (ndarray): The 2-D array to interpolate
        xCoords (ndarray): A 2-D array of x-coordinates, y-coordinates
        interpArray (ndarray): The array to place the calculation
    Returns:
        interpArray (ndarray): A pointer to the calculated ``interpArray''
    """
    if threads_per_block is None:
        threads_per_block = CUDA_TPB

    tpb = (threads_per_block,) * 3
    # blocks per grid
    bpg = (
            int(numpy.ceil(output_phase.shape[0] / threads_per_block)),
            int(numpy.ceil(output_phase.shape[1] / threads_per_block))
            )

    apply_correction_kernel[tpb, bpg](dm_correction, coords, output_phase, phs2rad)

    return output_phase

@cuda.jit
def apply_correction_kernel(dm_correction, coords, output_phase, phs2rad):
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
    correction_phase = 0
    if (i < output_phase.shape[0] and j < output_phase.shape[1]):
        for layer in range(dm_correction.shape[0]):
            # Get corresponding coordinates in image
            x = coords[layer, 0, i]
            if x >= dm_correction.shape[1] - 1:
                x = dm_correction.shape[1] - 1 - 1e-10
            x1 = numba.int32(x)
            y = coords[layer, 1, j]
            if y >= dm_correction.shape[2] - 1:
                y = dm_correction.shape[2] - 1 - 1e-10
            y1 = numba.int32(y)

            # Do bilinear interpolation
            xGrad1 = dm_correction[layer, x1+1, y1] - dm_correction[layer, x1, y1]
            a1 = dm_correction[layer, x1, y1] + xGrad1*(x-x1)

            xGrad2 = dm_correction[layer, x1+1, y1+1] - dm_correction[layer, x1, y1+1]
            a2 = dm_correction[layer, x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1

            phase_val = a1 + yGrad*(y-y1)
            correction_phase -= phase_val

        output_phase[i, j] = correction_phase * phs2rad
