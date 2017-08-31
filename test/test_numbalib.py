from soapy import numbalib
import numpy

import aotools

def test_zoomtoefield():
    """
    Checks that when zooming to efield, the same result is found as when zooming
    then using numpy.exp to get efield.
    """
    input_data = numpy.arange(100).reshape(10,10).astype("float32")

    output_data = numpy.zeros((100, 100), dtype="float32")
    output_efield2 = numpy.zeros((100, 100), dtype="complex64")
    

    numbalib.wfs.zoom(input_data, output_data)

    output_efield1 = numpy.exp(1j * output_data)

    numbalib.wfs.zoomtoefield(input_data, output_efield2)

    assert numpy.allclose(output_efield1, output_efield2)



def test_chop_subaps_mask():
    """
    Tests that the numba routing chops phase into sub-apertures in the same way
    as using numpy indices
    """
    nx_phase = 12
    nx_subap_size = 3
    nx_subaps = nx_phase // nx_subap_size

    phase = (numpy.random.random((nx_phase, nx_phase)) 
                + 1j * numpy.random.random((nx_phase, nx_phase))
                ).astype("complex64")
    subap_array = numpy.zeros((nx_subaps * nx_subaps, nx_subap_size, nx_subap_size)).astype("complex64")
    numpy_subap_array = subap_array.copy()

    mask = aotools.circle(nx_phase/2., nx_phase)

    x_coords, y_coords = numpy.meshgrid(
            numpy.arange(0, nx_phase, nx_subap_size),
            numpy.arange(0, nx_phase, nx_subap_size))
    subap_coords = numpy.array([x_coords.flatten(), y_coords.flatten()]).T

    numpy_chop(phase, subap_coords, nx_subap_size, numpy_subap_array, mask)
    numbalib.wfs.chop_subaps_mask(
            phase, subap_coords, nx_subap_size, subap_array, mask)
    assert numpy.array_equal(numpy_subap_array, subap_array)


def numpy_chop(phase, subap_coords, nx_subap_size, subap_array, mask):
    """
    Numpy vesion of chop subaps tests
    """
    mask_phase = phase * mask
    for n, (x, y) in enumerate(subap_coords):
        subap_array[n] = mask_phase[
                x: x + nx_subap_size,
                y: y + nx_subap_size
        ]
    return subap_array
    

def test_abs_squared():
    """
    Tests that the numba vectorised and parallelised abs squared gives the same result as numpy
    """
    data = (numpy.random.random((100, 20, 20))
            + 1j * numpy.random.random((100, 20, 20))).astype("complex64")

    output_data = numpy.zeros((100, 20, 20), dtype="float32")

    numbalib.abs_squared(data, out=output_data)

    assert numpy.array_equal(output_data, numpy.abs(data)**2)

def test_place_subaps_detector():

    nx_subaps = 4
    pxls_per_subap = 4
    tot_pxls_per_subap = 2 * pxls_per_subap # More for total FOV
    tot_subaps = nx_subaps * nx_subaps
    nx_pxls = nx_subaps * pxls_per_subap

    detector = numpy.zeros((nx_pxls, nx_pxls))
    detector_numpy = detector.copy()

    subaps = numpy.random.random((tot_subaps, tot_pxls_per_subap, tot_pxls_per_subap))
    # Find the coordinates of the vertices of the subap on teh detector
    detector_coords = []
    subap_coords = []
    for ix in range(nx_subaps):
        x1 = ix * pxls_per_subap - pxls_per_subap/2
        x2 = (ix + 1) * pxls_per_subap + pxls_per_subap/2
        sx1 = 0
        sx2 = tot_pxls_per_subap
        for iy in range(nx_subaps):
            y1 = iy * pxls_per_subap - pxls_per_subap/2
            y2 = (iy + 1) * pxls_per_subap + pxls_per_subap/2
            
            sy1 = 0
            sy2 = tot_pxls_per_subap
            # Check for edge subaps that would be out of bounds
            if x1 < 0:
                x1 = 0
                sx1 = tot_pxls_per_subap / 4
            if x2 > nx_pxls:
                x2 = nx_pxls
                sx2 = 3 * tot_pxls_per_subap / 4
            
            if y1 < 0:
                y1 = 0
                sy1 = tot_pxls_per_subap / 4
            if y2 > nx_pxls:
                y2 = nx_pxls
                sy2 = 3 * tot_pxls_per_subap / 4
            
            detector_coords.append(numpy.array([x1, x2, y1, y2]))
            subap_coords.append(numpy.array([sx1, sx2, sy1, sy2]))

    detector_coords = numpy.array(detector_coords).astype("int")
    subap_coords = numpy.array(subap_coords).astype("int")

    numbalib.wfs.place_subaps_on_detector(
        subaps, detector, detector_coords, subap_coords)

    numpy_place_subaps(subaps, detector_numpy, detector_coords, subap_coords)

    assert numpy.array_equal(detector, detector_numpy)

def numpy_place_subaps( subap_arrays, detector, detector_subap_coords, valid_subap_coords):
    
    for n, (x1, x2, y1, y2) in enumerate(detector_subap_coords):
        sx1, sx2, sy1, sy2 = valid_subap_coords[n]
        subap = subap_arrays[n]    
        detector[x1: x2, y1: y2] += subap[sx1: sx2, sy1: sy2]
    
    return detector


def test_geometric_propagation():

    phase_screens = numpy.random.random((10, 128, 128))

    phase_buf = numpy.zeros((128, 128))


    efield_buf = numpy.ones((128, 128), dtype="complex64")
    phase2Rad = 0.5
    propagation_direction = 1

    numbalib.los.makePhaseGeometric(phase_screens, phase2Rad, propagation_direction, phase_buf, efield_buf)

    phase1 = phase_buf.copy()
    efield1 = efield_buf.copy()

    phase_buf[:] = 0
    efield_buf[:] = 1
    numbalib.los.makePhaseGeometric_numpy(phase_screens, phase2Rad, propagation_direction, phase_buf, efield_buf)


    assert numpy.array_equal(phase1, phase_buf)
    assert numpy.array_equal(efield1, efield_buf)

def test_perform_correction():

    correction_screens = numpy.random.random((10, 128, 128))

    phase_buf = numpy.zeros((128, 128))
    residual_buf = numpy.zeros_like(phase_buf)
    phase_correction = phase_buf.copy()

    efield_buf = numpy.ones((128, 128), dtype="complex64")
    phase2Rad = 0.5
    propagation_direction = -1

    residual_buf = numbalib.los.perform_correction(
            correction_screens, phase2Rad, propagation_direction, phase_correction,
            phase_buf, efield_buf, residual_buf)

    phase1 = phase_buf.copy()
    efield1 = efield_buf.copy()
    residual1 = residual_buf.copy()
    phase_correction1 = phase_correction.copy()

    residual_buf[:] = 0
    phase_buf[:] = 0
    efield_buf[:] = 1
    phase_correction[:] = 0

    residual_buf = numbalib.los.perform_correction_numpy(
            correction_screens, phase2Rad, propagation_direction, phase_correction,
            phase_buf, efield_buf, residual_buf)

    assert numpy.array_equal(phase_correction1, phase_correction)
    assert numpy.array_equal(phase1, phase_buf)
    assert numpy.array_equal(efield1, efield_buf)
    assert numpy.array_equal(residual1, residual_buf)


if __name__ == "__main__":
    test_zoomtoefield()
