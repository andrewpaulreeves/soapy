import numpy
from numba import cuda

from soapy import gpulib, numbalib

def test_apply_correction():

    correction = numpy.ones((2, 128, 128), dtype="float32")
    correction_gpu = cuda.to_device(correction)

    phase = numpy.zeros((128, 128), dtype="float32")
    phase_gpu = cuda.to_device(phase)

    coords = numpy.zeros((2, 2, 128))
    for i in range(2):
        coords[i, :] = numpy.linspace(32, 96-1, 128).astype('float32')
    coords_gpu = cuda.to_device(coords)

    # Do GPU computation
    gpulib.los.apply_correction(correction_gpu, coords_gpu, phase_gpu, phs2rad=1)
    gpu_phase = phase_gpu.copy_to_host()

    # Do CPU computation
    phase_cpu = phase.copy()
    thread_pool = numbalib.ThreadPool(1)
    interp_correction = numpy.zeros_like(phase)
    for i in range(len(correction)):
        interp_correction[:] = 0
        interp_correction = numbalib.los.bilinear_interp(
                correction[i], coords[i, 0], coords[i, 1], interp_correction, thread_pool=thread_pool)
        phase_cpu -= interp_correction

    assert numpy.allclose(phase_cpu, gpu_phase)

if __name__ == "__main__":
    test_apply_correction()