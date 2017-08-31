import numpy
import numba
from . import numbalib

def makePhaseGeometric_numpy(phase_screens, phase2Rad, propagation_direction, phase_array, EField_array):
    '''
    Creates the total phase along line of sight offset by a given angle using a geometric ray tracing approach

    Parameters:
        phase_screens (ndarray): 3-D array of phase screens
        phase2Rad (float): conversion of phase in nm to radians
        propagation_direction (int): 1 if propagating down, -1 if up
        phase_array (ndarray): Buffer to place propagated phase
        EField_array (ndarray): Complex Buffer to place propagated EField

    Returns:
        ndarray: View of EField buffer
    '''

    phase_screens.sum(0, out=phase_array)

    # Convert phase to radians
    phase_array *= phase2Rad

    # Change sign if propagating up
    phase_array *= propagation_direction

    numpy.exp(1j * phase_array, out=EField_array)

    return EField_array


@numba.jit(nopython=True, parallel=True, nogil=True)
def makePhaseGeometric(phase_screens, phase2Rad, propagation_direction, phase_array, EField_array):
    '''
    Creates the total phase along line of sight offset by a given angle using a geometric ray tracing approach

    Parameters:
        phase_screens (ndarray): 3-D array of phase screens
        phase2Rad (float): conversion of phase in nm to radians
        propagation_direction (int): 1 if propagating down, -1 if up
        phase_array (ndarray): Buffer to place propagated phase
        EField_array (ndarray): Complex Buffer to place propagated EField

    Returns:
        ndarray: View of EField buffer
    '''

    # Loop over the layers
    for l in range(phase_screens.shape[0]):
        # Loop over x vals in screen
        for x in numba.prange(phase_screens.shape[1]):
            # Loop over y vals in screen
            for y in range(phase_screens.shape[2]):
                phase_array[x, y] += (phase_screens[l, x, y] * phase2Rad * propagation_direction)

    # Loop over x and y again to convert phase to EField
    for x in numba.prange(phase_array.shape[0]):
        for y in range(phase_array.shape[1]):
            EField_array[x, y] = numpy.exp(1j * phase_array[x, y])

    return EField_array


def perform_correction_numpy(
            correction_screens, phase2Rad, propagation_direction, phase_correction,
            phase_array, EField_array, residual):


    correction_screens.sum(0, out=phase_correction)

    # Correct EField
    EField_array *= numpy.exp(propagation_direction * 1j * phase_correction * phase2Rad)

    # Also correct phase in case its required
    residual = phase_array / phase2Rad - phase_correction

    phase_array[:] = residual * phase2Rad

    return residual

@numba.jit(nopython=True, parallel=True)
def perform_correction(
        correction_screens, phase2Rad, propagation_direction, phase_correction,
        phase_array, EField_array, residual):

    for dm in range(correction_screens.shape[0]):
        for x in numba.prange(correction_screens.shape[1]):
            for y in range(correction_screens.shape[2]):

                phase_correction[x, y] += (correction_screens[dm, x, y])

    for x in numba.prange(phase_array.shape[0]):
        for y in range(phase_array.shape[1]):
            EField_array[x, y] *= numpy.exp(propagation_direction * 1j * phase_correction[x, y] * phase2Rad)
            residual[x, y] = (phase_array[x, y] / phase2Rad) - phase_correction[x, y]
            phase_array[x, y] = residual[x, y] * phase2Rad

    return residual