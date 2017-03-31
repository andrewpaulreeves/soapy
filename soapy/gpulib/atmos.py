import math

import numpy
import numba
from numba import cuda

from . import gpulib
def interp_phase(screen, output_screen, interp_coords, float_position, threads_per_block=None, stream=None):
    if threads_per_block is None:
        threads_per_block = gpulib.CUDA_TPB

    tpb = (threads_per_block, threads_per_block)
    bpg = (int(numpy.ceil(float(output_screen.shape[0]) / threads_per_block)),
           int(numpy.ceil(float(output_screen.shape[1]) / threads_per_block)))

    interp_phase_kernel[tpb, bpg, stream](screen, output_screen, interp_coords, float_position)

    return output_screen

@cuda.jit
def interp_phase_kernel(screen, output_screen, interp_coords, float_position):

    i, j = cuda.grid(2)

    # Check bounds
    if i<output_screen.shape[0] and j<output_screen.shape[1]:
        x = interp_coords[i] - float_position
        y = interp_coords[j]

        x_int = numba.int32(x)
        y_int = numba.int32(y)

        x_grad1 = screen[x_int+1, y_int] - screen[x_int, y_int]
        a1 = screen[x_int, y_int] + x_grad1*(x-x_int)

        x_grad2 = screen[x_int+1, y_int+1] - screen[x_int, y_int+1]
        a2 = screen[x_int, y_int+1] + x_grad2*(x-x_int)

        y_grad = a2 - a1
        output_screen[i, j] = a1 + y_grad*(y - y_int)


def get_phase_points(data, output_data, coordinates, threads_per_block=None, stream=None):

    if threads_per_block is None:
        threads_per_block = gpulib.CUDA_TPB

    tpb = threads_per_block
    # blocks per grid
    bpg = int(numpy.ceil(float(coordinates.shape[0]) / threads_per_block))

    get_phase_points_kernel[tpb, bpg, stream](data, output_data, coordinates)

    return data

@cuda.jit
def get_phase_points_kernel(data, output_data, coordinates):
    i = cuda.grid(1)

    if i < coordinates.shape[0]:
        x = coordinates[i, 0]
        y = coordinates[i, 1]

        output_data[i] = data[x, y]


def add_row(screen, new_row, threads_per_block=None, stream=None):
    if threads_per_block is None:
        threads_per_block = gpulib.CUDA_TPB

    tpb = (threads_per_block, threads_per_block)
    bpg = (int(numpy.ceil(float(screen.shape[0]) / threads_per_block)),
           int(numpy.ceil(float(screen.shape[1]) / threads_per_block)))


    add_row_kernel[tpb, bpg, stream](screen, new_row)

    return screen

@cuda.jit
def add_row_kernel(screen, new_row):
    i, j = cuda.grid(2)

    # Check in bounds
    if (i < screen.shape[0]) and (j < screen.shape[1]):
        # if in first row of screen, will use the new data
        if i == 0:
            new_screen_val = new_row[j]

        # Else, will use the next along bit of old screen
        else:
            new_screen_val = screen[i-1, j]

        # Wait for everyone to get here so we don't overwrite stuff we want to read!
        cuda.syncthreads()

        # Can now safely assign the new value to the screen array
        screen[i, j] = new_screen_val

def get_subscreen(screen, sub_screen, offset=None, threads_per_block=None, stream=None):

    if threads_per_block is None:
        threads_per_block = gpulib.CUDA_TPB

    if offset is None:
        offset = numpy.array([0, 0])

    tpb = (threads_per_block, threads_per_block)
    bpg = (int(numpy.ceil(float(sub_screen.shape[0]) / threads_per_block)),
           int(numpy.ceil(float(sub_screen.shape[1]) / threads_per_block)))

    get_subscreen_kernel[tpb, bpg, stream](screen, sub_screen, offset)

    return sub_screen

@cuda.jit
def get_subscreen_kernel(screen, sub_screen, offset):

    i, j = cuda.grid(2)
    if (i < sub_screen.shape[0]) and (j < sub_screen.shape[1]):

        sub_screen[i, j] = screen[i + offset[0], j + offset[1]]

def gather_screens(screens_buffer, screens, wavelength_multiplier, threads_per_block=None, stream=None):
    if threads_per_block is None:
        threads_per_block = gpulib.CUDA_TPB

    tpb = (threads_per_block, threads_per_block)
    bpg = (int(numpy.ceil(float(screens_buffer.shape[1]) / threads_per_block)),
           int(numpy.ceil(float(screens_buffer.shape[2]) / threads_per_block)))

    for screen_no, screen in enumerate(screens):
        gather_screens_kernel[tpb, bpg, stream](screens_buffer, screen, wavelength_multiplier, screen_no)

    return screens_buffer

@cuda.jit
def gather_screens_kernel(screens_buffer, screen, wavelength_multiplier, n_screen):

    i, j = cuda.grid(2)

    if i < screen.shape[0] and j < screen.shape[1]:
        screens_buffer[n_screen, i, j] = screen[i, j] * wavelength_multiplier