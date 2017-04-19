class Plugin(TransformPlugin):

    def __init__(self):
        TransformPlugin.__init__(self, params={'base (not currently implemented)':10},
                name='Log')

    def transform(self):
        if len(self.data) == 2:
            return numpy.array([self.data[0],numpy.log10(self.data[1])]).T
        else:
            return numpy.log(self.data)
