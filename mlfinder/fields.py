# basic imports
import numpy as np
import pandas as pd

import astropy
from astropy.table import Table

# import from module
from mlfinder.bd import BrownDwarf

# class for the fields (potentially either many or one)
class Fields():
    def __init__(self, which=None, ra = None, dec = None, bd = None):
        self.which = which
        
        # brown dwarf can be ra/dec or data or class
        if ra is not None and dec is not None:
            self.ra = ra
            self.dec = dec
        
        elif isinstance(bd, (Table, pd.DataFrame)):
            # first change df into what want
            bd = find_info(bd)
            
            # basic info
            self.ra = bd['ra']
            self.dec = bd['dec']
            self.mu_a = bd['mu_alpha']
            self.mu_d = bd['mu_delta']
            self.pi = bd['pi']
        
        elif isinstance(bd, BrownDwarf):
            self.ra = bd.ra
            self.dec = bd.dec
            
        else:
            raise Exception('Brown Dwarf data needs to either be ra/dec, an astropy table or pandas table of the dwarf data, or the brown dwarf class.')
            
        # now to grab the star info
        if which is None:
            print('No field selected. Looking through them all.')
            
            self.fields = [] #ADD
            
        else:
            self.fields = which
            
        # grabbing stars themselves
        # self.stars = 
        
        
        self.file = r'C:\Users\judah\candidate_stars_ephemerides\dr8\0855-0714_bs.txt'
        
        self.stars = pd.read_csv(self.file)
        self.stars = self.filter_stars_only(self.stars)
        self.stars = self.filter_stars_mag(self.stars)
     
    #def grab_stars():
    #    return None

    ##
    # Name: filter_stars_only
    #
    # input: dataframe of background objects
    # output: modified dataframe of just background stars
    #
    # purpose: filter out objects in dataframe that aren't stars (like galaxies) 
    #
    def filter_stars_only(self, dataset):
        drop_l = list()
        for i in range(len(dataset['type'])):
            if dataset['type'][i] != 'PSF':
                drop_l.append(i)

        return dataset.drop(drop_l, axis=0) 
    
    ##
    # Name: filter_stars_mag
    #
    # input: dataframe of background stars
    # output: modified dataframe of background stars
    #
    # purpose: filter out stars don't know mag_r of and dimmer than DECaLS PSF limit
    #
    def filter_stars_mag(self, stars):
        mags = list()

        #filtering NaN, Infinity, and 0<=star<=30 values
        df = pd.DataFrame(stars)

        df = df.replace('Infinity', np.nan) #replace infinities with nan
        df = df.dropna(subset=['dered_mag_r']) #drop nan

        df['dered_mag_r'] = pd.to_numeric(df['dered_mag_r'])

        df = df[df.dered_mag_r >= 0]
        df= df[df.dered_mag_r <= 23.54]

        return df
