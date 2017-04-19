"""
Example Analysis Plugin - average
_________________________________

This simple plugin takes the average of whatever data is plotted in the
MagicPlot window.
"""
class Plugin(AnalysisPlugin):

    def __init__(self):
        """
        Set the ``params`` and ``name``, here there are no parameters required
        so we set params to an empty dict.
        """
        AnalysisPlugin.__init__(self, params={},
            name='Average')

    def run(self):
        """
        Return the average of ``self.data``.
        """
        return numpy.average(self.data)

