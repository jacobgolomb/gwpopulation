from .models.mass import *
from .models.redshift import *

class DetectorFramePopulationProbability(object):
    def __init__(mass_model, redshift_model, mass_kwargs={}, redshift_kwargs={}):

        self.mass_model = mass_model(**mass_kwargs)
        self.redshift_model = redshift_model(**redshift_kwargs)
