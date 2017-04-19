class Plugin(TransformPlugin):

    def __init__(self):
        TransformPlugin.__init__(self, params={'add':1},
                name='add')

    def transform(self):
        # 1D data has length 2, as in (x,y)
        # Stupidly pyqtgraph can't plot data in the format it gives you, so 
        # need to give it an array of shape (N, 2) with x and y columns
        if len(self.data) == 2:
            return numpy.array([self.data[0], self.data[1] + self.params['add']]).T
        else:
            return self.data + self.params['add']
