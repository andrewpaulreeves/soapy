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
        interp_correction = numbalib.bilinear_interp(
                correction[i], coords[i, 0], coords[i, 1], interp_correction, thread_pool=thread_pool)
        phase_cpu -= interp_correction

    assert numpy.allclose(phase_cpu, gpu_phase)

def test_add_row():

    data = numpy.arange(100).reshape(10,10).astype("float32")
    new_data = numpy.arange(-10, 0).astype("float32")

    # Data with a blanck space at the beginning
    gpu_data = cuda.to_device(numpy.append(data, numpy.zeros((1, 10)),  axis=0))

    # CPU append
    append_data_cpu = numpy.append(new_data.reshape(1, 10), data, axis=0)

    # GPU append
    gpulib.atmos.add_row(gpu_data, cuda.to_device(new_data))

    assert numpy.array_equal(append_data_cpu, gpu_data.copy_to_host())



if __name__ == "__main__":
    test_apply_correction()
    test_add_row()