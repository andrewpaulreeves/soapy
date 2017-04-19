from scipy.stats import linregress
class Plugin(AnalysisPlugin):

    def __init__(self):
        AnalysisPlugin.__init__(self, params={},
                name='Gradient')

    def run(self):
        if len(self.data) !=2:
            raise Exception('Only works with 1D plots')

        ################################################
        slope, intercept, rvalue, pvalue, stderr = linregress(
                self.data[0], self.data[1])
        ##################################################

        return {'Gradient': slope, 'Intercept': intercept,
                'Standard Error': stderr}
