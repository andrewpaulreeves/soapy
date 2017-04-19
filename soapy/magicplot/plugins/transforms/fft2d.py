class Plugin(TransformPlugin):

    def __init__(self):
        TransformPlugin.__init__(self, params={},
                name='2d Fourier Transform')

    def transform(self):
        if len(self.data) == 2:
            raise('Only works with 2D plts')
        else:
            return abs(numpy.fft.fftshift(numpy.fft.fft2(self.data)))
