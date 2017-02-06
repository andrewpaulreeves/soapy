import multiprocessing
N_CPU = multiprocessing.cpu_count()
from threading import Thread
from queue import Queue
import logging
import time

import numpy
import numba
import pyfftw

from aotools import wfs as wfslib


ASEC2RAD = (numpy.pi/(180 * 3600))

class ShackHartmann2(object):
    """
    A Shack-Hartmann WFS

    An object that accepts a phase instance, and calculates the resulting measurements as would be observed by a Shack-Hartmann WFS. To do this, the phase is first interpolated linearly to make a size that will return the desired pixel scale. The phase is then split into sub-apertures, each of which is brought to a focal plane using an FFT. The individual sub-apertures are placed back into a SH WFS detector image. A centroiding algorithm is then performed on each sub-aperture this detector array

    Parameters:
        soapy_config (SoapyConfig):
        wfs_index (int):
        mask (ndarray):
        los (LineOfSight, optional): Corresponding Soapy Line of sight object. Can be given so its easier for other modules to retrieve it

    """
    def __init__(self, soapy_config, wfs_index, mask=None, los=None):

        self.los = los

        # Get parameters from configuration
        # ---------------------------------
        wfs_config = soapy_config.wfss[wfs_index]

        self.pupil_size = soapy_config.sim.pupilSize
        self.threads = soapy_config.sim.threads

        self.mask = mask[
                soapy_config.sim.simPad:-soapy_config.sim.simPad,
                soapy_config.sim.simPad:-soapy_config.sim.simPad
                ]
        self.telescope_diameter = soapy_config.tel.telDiam

        self.subap_fov = wfs_config.subapFOV
        self.nx_subaps = wfs_config.nxSubaps
        self.wavelength = wfs_config.wavelength
        self.nx_subap_pxls = wfs_config.pxlsPerSubap
        self.subap_threshold = wfs_config.subapThreshold


        # Calculate some parameters
        # -------------------------
        self.pxl_scale = self.subap_fov/self.nx_subap_pxls
        self.subap_diam = self.telescope_diameter/self.nx_subaps

        # coordinates of sub-ap positions on pupil and detector
        self.subap_positions, self.subapFillFactor = wfslib.findActiveSubaps(
                self.nx_subaps, self.mask,
                self.subap_threshold, returnFill=True)
        self.subap_detector_pos = self.subap_positions * self.nx_subaps * self.nx_subap_pxls / self.pupil_size
        self.slope_calc_coords = self.subap_detector_pos.copy()
        self.subap_detector_pos = numpy.array([
                self.subap_detector_pos[:, 0], self.subap_detector_pos[:, 0] + self.nx_subap_pxls,
                self.subap_detector_pos[:, 1], self.subap_detector_pos[:, 1] + self.nx_subap_pxls
                    ]).T

        self.n_subaps = self.subap_positions.shape[0]
        self.nx_subap_size = float(self.pupil_size)/self.nx_subaps # Subap diameter in phase elements
        self.n_measurements = self.n_subaps * 2 # Number of total measurements

        # size of each sub-ap before FFT
        self.nx_subap_interp = int(round(
                self.nx_subap_size * self.pxl_scale * (numpy.pi/(180*3600)) * self.subap_diam
                / self.wavelength))
        print("Active Subapertures: {}".format(self.n_subaps))
        print("Subap phase elements: {}".format(self.nx_subap_size))
        print("nx_subap_interp: {}".format(self.nx_subap_interp))

        # Find the sub-ap coordinates on the interpolated phase map
        self.subap_interp_positions = numpy.round(
                self.subap_positions * (float(self.nx_subap_interp)/self.nx_subap_size)
                ).astype('int32')

        # Find first multiple of nx_subap_pxls bigger than nx_subap_interp
        self.detector_bin_ratio = 0
        self.nx_subap_focus_efield = 1
        while self.nx_subap_focus_efield < self.nx_subap_interp:
            self.detector_bin_ratio += 1
            self.nx_subap_focus_efield = self.nx_subap_pxls * self.detector_bin_ratio
        print("Set detector bin ratio to {}, nx_subap_focus_efield: {}".format(
                self.detector_bin_ratio, self.nx_subap_focus_efield))

        # make an fft object
        self.subap_efield = pyfftw.zeros_aligned(
                (self.n_subaps, self.nx_subap_focus_efield, self.nx_subap_focus_efield), 
                dtype="complex64")
        self.subap_focus_efield = self.subap_efield.copy()
        self.fft = pyfftw.FFTW(
                self.subap_efield, self.subap_focus_efield, 
                threads=self.threads, axes=(1,2))

        # Number of pixels on detector
        self.nx_detector_pxls = self.nx_subaps * self.nx_subap_pxls

        # init a centroider
        self.centroider = Centroider(self.n_subaps, self.nx_subap_pxls, threads=self.threads)

        # Calculate a tilt required to centre the spots
        self.tilt_fix = self.calculate_tilt_correction()

        # Find rad to nm conversion
        self.nm_to_rad = 1e-9 * (2 * numpy.pi) / self.wavelength

        # Array place holders
        self.subap_phase = numpy.zeros(
                (self.n_subaps, self.nx_subap_interp, self.nx_subap_interp), dtype="float32")
        self.phase = numpy.zeros((self.pupil_size, self.pupil_size), dtype="float32")
        self.interp_phase = numpy.zeros((self.nx_subaps * self.nx_subap_interp, self.nx_subaps * self.nx_subap_interp), dtype='float32')
        self.slopes = numpy.zeros(2 * self.subap_positions.shape[0], dtype="float32")
        self.detector_subaps = numpy.zeros(
                (self.n_subaps, self.nx_subap_pxls, self.nx_subap_pxls))
        self.detector = numpy.zeros(
            (self.nx_detector_pxls, self.nx_detector_pxls), dtype="float32")
        self.slope_calc_subaps = numpy.zeros(
                (self.n_subaps, self.nx_subap_pxls, self.nx_subap_pxls))
        self.subaps_focus_intensity = numpy.zeros(
                (self.n_subaps, self.nx_subap_focus_efield, self.nx_subap_focus_efield),
                dtype="float32")

        # Run the WFS once with zero phase to get static slopes
        self.static_slopes = numpy.zeros_like(self.slopes)
        self.static_slopes = self.frame(numpy.zeros((self.pupil_size, self.pupil_size)))

    def calculate_tilt_correction(self):
        """
        Calculates the required tilt to add to avoid the PSF being centred on
        only 1 pixel
        """
        if not self.nx_subap_pxls % 2:
            # If pixels per subap is even
            # Angle we need to correct for half a pixel
            theta = ASEC2RAD *  self.pxl_scale * self.nx_subap_pxls/ (2*self.nx_subap_focus_efield)

            # Magnitude of tilt required to get that angle
            A = theta * self.telescope_diameter/(2*self.wavelength)*2*numpy.pi

            # Create tilt arrays and apply magnitude
            coords = numpy.linspace(-1, 1, self.pupil_size)
            X, Y = numpy.meshgrid(coords, coords)

            tilt_fix = -1 * A * (X + Y)

        else:
            tilt_fix = numpy.zeros((self.nx_subap_interp,)*2)

        return tilt_fix

    def interp_to_size(self):
        """
        Interpolate the phase to the size required for the correct pixel scale
        """
        # print("Interp")
        # Convert to rad
        self.phase *= self.nm_to_rad

        # add tilt fix
        self.phase += self.tilt_fix

        zoom(self.phase, self.interp_phase, threads=self.threads)

    def chop_into_subaps(self):
        """
        Chop the phase into individual sub-apertures and turns each into a complex amplitude
        """
        # print("chop")
        chop_subaps_efield(
                self.interp_phase, self.subap_interp_positions,
                self.nx_subap_interp, self.subap_efield, threads=self.threads)

    def subaps_to_focus(self):
        """
        Bring each of the sub-aperture's phase to a focus
        """
        # print("focus")
        self.fft()

        abs_squared(self.subap_focus_efield, self.subaps_focus_intensity, threads=self.threads)
        self.subaps_focus_intensity = numpy.fft.fftshift(self.subaps_focus_intensity, axes=(1,2))

    def assemble_detector_image(self):
        """
        Bin focus to detector pixel scale and put into a detector image
        """
        # print("assemble detector")
        if self.nx_subap_focus_efield != self.nx_subap_pxls:
            bin_imgs(
                    self.subaps_focus_intensity, self.detector_bin_ratio, 
                    self.detector_subaps, threads=self.threads)
        else:
            self.detector_subaps = self.subaps_focus_intensity

        # put subaps back into a single detector array
        place_subaps_on_detector(
                self.detector_subaps, self.detector, self.subap_detector_pos,
                threads=self.threads)

    def calculate_slopes(self):
        """
        Given a SH WFS detector image, will compute the centroid slope values
        """
        # print("calc_slopes")
        chop_subaps(
                self.detector, self.slope_calc_coords,
                self.nx_subap_pxls, self.slope_calc_subaps, threads=self.threads)

        # print("Do Centroider")
        self.slopes = self.centroider(self.slope_calc_subaps).T.flatten()

        # correct for static slopes
        self.slopes -= self.static_slopes

    def frame(self, phase, read=False):

        # convert phase to radians
        self.phase = phase

        self.interp_to_size()
        
        self.chop_into_subaps()

        self.subaps_to_focus()

        self.assemble_detector_image()

        self.calculate_slopes()

        return self.slopes
    

# WFS Library Functions
# ---------------------

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


    if threads is None:
        threads = N_CPU

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

    for i in range(chunkIndices[0], chunkIndices[1]):
        x = i*numba.float32(data.shape[0]-1)/(zoomArray.shape[0]-0.99999999)
        x1 = numba.int32(x)
        for j in range(zoomArray.shape[1]):
            y = j*numba.float32(data.shape[1]-1)/(zoomArray.shape[1]-0.99999999)
            y1 = numba.int32(y)

            xGrad1 = data[x1+1, y1] - data[x1, y1]
            a1 = data[x1, y1] + xGrad1*(x-x1)

            xGrad2 = data[x1+1, y1+1] - data[x1, y1+1]
            a2 = data[x1, y1+1] + xGrad2*(x-x1)

            yGrad = a2 - a1
            zoomArray[i,j] = a1 + yGrad*(y-y1)


    return zoomArray

def chop_subaps(phase, subap_coords, nx_subap_size, subap_array, threads=None):

    if threads is None:
        threads = N_CPU

    n_subaps = subap_coords.shape[0]
    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=chop_subaps_numba,
            args = (
                phase, subap_coords, nx_subap_size, subap_array,
                numpy.array([int(t*n_subaps/threads), int((t+1)*n_subaps/threads)]),
                )
            ))
        Ts[t].start()

    for T in Ts:
        T.join()
    
    return subap_array

@numba.jit(nopython=True, nogil=True)
def chop_subaps_numba(phase, subap_coords, nx_subap_size, subap_array, subap_indices):

    for i in range(subap_indices[0], subap_indices[1]):
            x = subap_coords[i, 0]
            y = subap_coords[i, 1]

            subap_array[i] = phase[x:x+nx_subap_size, y:y+nx_subap_size]
            
    return subap_array

def chop_subaps_slow(phase, subap_coords, nx_subap_size, subap_array, threads=None):


    for i in range(len(subap_coords)):
            x = subap_coords[i, 0]
            y = subap_coords[i, 1]

            subap_array[i] = phase[x: x + nx_subap_size, y: y + nx_subap_size]
            
    return subap_array


def chop_subaps_efield(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_coords.shape[0]
    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=chop_subaps_efield_numba,
                         args=(
                             phase, subap_coords, nx_subap_size, subap_array,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return subap_array


@numba.jit(nopython=True, nogil=True)
def chop_subaps_efield_numba(phase, subap_coords, nx_subap_size, subap_array, subap_indices):
    for i in range(subap_indices[0], subap_indices[1]):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = numpy.exp(1j * phase[x:x + nx_subap_size, y:y + nx_subap_size])

    return subap_array


def chop_subaps_efield_slow(phase, subap_coords, nx_subap_size, subap_array, threads=None):
    for i in range(len(subap_coords)):
        x = subap_coords[i, 0]
        y = subap_coords[i, 1]

        subap_array[i, :nx_subap_size, :nx_subap_size] = numpy.exp(1j * phase[x: x + nx_subap_size, y: y + nx_subap_size])

    return subap_array


def place_subaps_on_detector(subap_imgs, detector_img, subap_positions, threads=None):

    if threads is None:
        threads = N_CPU

    n_subaps = subap_positions.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=place_subaps_on_detector_numba,
            args = (
                subap_imgs, detector_img, subap_positions,
                numpy.array([int(t*n_subaps/threads), int((t+1)*n_subaps/threads)]),
                )
            ))
        Ts[t].start()

    for T in Ts:
        T.join()
    
    return detector_img

@numba.jit(nopython=True, nogil=True)
def place_subaps_on_detector_numba(subap_imgs, detector_img, subap_positions, subap_indices):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_indices[0], subap_indices[1]):
        x1, x2, y1, y2 = subap_positions[i]

        detector_img[x1: x2, y1: y2] = subap_imgs[i]

    return detector_img

def place_subaps_on_detector_slow(subap_imgs, detector_img, subap_positions, threads=None):
    """
    Puts a set of sub-apertures onto a detector image
    """

    for i in range(subap_positions.shape[0]):
        x1, x2, y1, y2 = subap_positions[i]

        detector_img[x1: x2, y1: y2] = subap_imgs[i]

    return detector_img


def bin_imgs(subap_imgs, bin_size, binned_imgs, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_imgs.shape[0]


    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=bin_imgs_numba,
                         args=(
                             subap_imgs, bin_size, binned_imgs,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )
                         ))
        Ts[t].start()

    for T in Ts:
        T.join()

    return binned_imgs


@numba.jit(nopython=True, nogil=True)
def bin_imgs_numba(imgs, bin_size, new_img, subap_range):

    # loop over subaps
    for n in range(subap_range[0], subap_range[1]):
        # loop over each element in new array
        for i in range(new_img.shape[1]):
            x1 = i * bin_size
            for j in range(new_img.shape[2]):
                y1 = j * bin_size
                new_img[n, i, j] = 0
                # loop over the values to sum
                for x in range(bin_size):
                    for y in range(bin_size):
                        new_img[n, i, j] += imgs[n, x + x1, y + y1]


def bin_imgs_slow(imgs, bin_size, new_img):
    # loop over subaps
    for n in range(imgs.shape[0]):
        # loop over each element in new array
        for i in range(new_img.shape[1]):
            x1 = i * bin_size
            for j in range(new_img.shape[2]):
                y1 = j * bin_size
                new_img[n, i, j] = 0
                # loop over the values to sum
                for x in range(bin_size):
                    for y in range(bin_size):
                        new_img[n, i, j] += imgs[n, x + x1, y + y1]

class Centroider(object):
    def __init__(self, n_subaps, nx_subap_pxls, threads=None):

        if threads is None:
            self.threads = 1
        else:
            self.threads = threads

        self.n_subaps = n_subaps
        self.nx_subap_pxls = nx_subap_pxls

        self.indices = numpy.indices((self.nx_subap_pxls, self.nx_subap_pxls))

        self.centroids = numpy.zeros((n_subaps, 2))

    def __call__(self, subaps):
        self.centre_of_gravity_numpy(subaps)
        return self.centroids


    def centre_of_gravity_numpy(self, subaps):
        self.centroids[:, 0] = ((self.indices[0]*subaps).sum((1,2))/subaps.sum((1,2))) + 0.5 - subaps.shape[1] * 0.5
        self.centroids[:, 1] = ((self.indices[1]*subaps).sum((1,2))/subaps.sum((1,2))) + 0.5 - subaps.shape[2] * 0.5
        return self.centroids

    def centre_of_gravity_numba(self, subaps):

        centre_of_gravity(subaps, self.indices, self.centroids, self.threads)
        return self.centroids

def centre_of_gravity(subaps, indices, centroids, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subaps.shape[0]

    Ts = []
    for t in range(threads):
        Ts.append(Thread(target=centre_of_gravity_numba,
                         args=(
                             subaps, indices, centroids,
                             numpy.array([int(t * n_subaps / threads), int((t + 1) * n_subaps / threads)]),
                         )))
        Ts[t].start()

    for T in Ts:
        T.join()

    return centroids


@numba.jit(nopython=True, nogil=True)
def centre_of_gravity_numba(subaps, indices, centroids, thread_indices):

    s1, s2 = thread_indices
    nx_subap_size = subaps.shape[1]
    subaps = subaps[s1:s2]

    centroids[s1:s2, 0] = (
            indices[0]*subaps).sum((1,2))/subaps.sum((1,2)) + 0.5 - nx_subap_size*0.5
    centroids[s1:s2, 1] = (
            indices[1]*subaps).sum((1,2))/subaps.sum((1,2)) + 0.5 - nx_subap_size*0.5


def abs_squared(subap_data, subap_output, threads=None):
    if threads is None:
        threads = N_CPU

    n_subaps = subap_data.shape[0]

    Ts = []
    for t in range(threads):
        x1 = int(t * n_subaps / threads)
        x2 = int((t+1) * n_subaps / threads)
        Ts.append(Thread(target=abs_squared_numba,
                         args=(
                             subap_data, subap_output,
                             numpy.array([x1, x2]),
                         )))

        Ts[t].start()

    for T in Ts:
        T.join()

    return subap_output

@numba.jit(nopython=True)
def abs_squared_numba(data, output_data, indices):

    for n in range(indices[0], indices[1]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                output_data[n, x, y] = data[n, x, y].real**2 + data[n, x, y].imag**2


def abs_squared_slow(data, output_data, threads=None):

    for n in range(data.shape[0]):
        for x in range(data.shape[1]):
            for y in range(data.shape[2]):
                output_data[n, x, y] = data[n, x, y].real**2 + data[n, x, y].imag**2


def run_threaded_func(func, args, error_queue):
    pass

def loop_sh(sh, phs, N=1000):
    t1 = time.time()
    for i in range(N):
        sh.frame(phs)
    t2 = time.time()
    f_time = (t2 - t1)/N
    print("Frame runs in {:.2f}ms".format(f_time*1000))
    print("Iters per second: {:2.2f}".format(1./f_time))


if __name__ == "__main__":

    wfs_config = WFS_Config()

    wfs_config.pupil_size = 640
    wfs_config.nx_subaps = 80
    wfs_config.subap_diam = 38./wfs_config.nx_subaps
    wfs_config.pxl_scale = 0.2
    wfs_config.nx_subap_pxls = 10
    wfs_config.wavelength = 500e-9
    X, Y = numpy.meshgrid(
            numpy.arange(0, wfs_config.pupil_size, wfs_config.pupil_size/wfs_config.nx_subaps), 
            numpy.arange(0, wfs_config.pupil_size, wfs_config.pupil_size/wfs_config.nx_subaps))
    wfs_config.subap_positions = numpy.array([X.flatten(), Y.flatten()]).T.astype('float32')    

    wfs_config.threads = 8

    phs = numpy.ones((wfs_config.pupil_size, wfs_config.pupil_size)).astype('float32')

    sh = ShackHartmann(wfs_config)
    sh.frame(phs)