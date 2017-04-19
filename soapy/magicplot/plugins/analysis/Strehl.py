from scipy.stats import linregress
class Plugin(AnalysisPlugin):

    def __init__(self):
        AnalysisPlugin.__init__(self, params={'wavelength(um)':1.65, 'pixel_scale':1, 
                'diameter':4.2,
                'central_obsc':0},
                name='Strehl')

    def run(self):
        # () The normalisation - the expected 
        norm = ( 3.1415926/4.*(1.-self.params['central_obsc']**2) /
                (0.206265*self.params['wavelength(um)']/self.params['diameter']/self.params['pixel_scale'] )**2 )
        # () The total integral under the recorded PSF:
        if len(self.data) == 2:
            image = self.data[1]
        else:
            image = self.data
        somme = image.sum()
        # () The maximum hight of the recorded PSF:
        psf_max_value =image.max()
        # () Calculate the Strehl ratio:
        SR = 100.*psf_max_value/(norm*somme)
        # () Return the Strehl ratio:
        return SR
