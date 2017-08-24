import numpy
import numba


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


