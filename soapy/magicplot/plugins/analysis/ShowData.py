class Plugin(AnalysisPlugin):

    def __init__(self):
        AnalysisPlugin.__init__(self, name='Data')

    def run(self):
        return self.data
