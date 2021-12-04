# basic imports
import numpy as np
import pandas as pd

import astropy
from astropy.table import Table

# imports from datalab (installation: https://datalab.noao.edu/docs/manual/UsingTheNOAODataLab/InstallDatalab/InstallDatalab.html)
import dl
from dl import queryClient as qc
from dl.helpers.utils import convert

# import from module
from mlfinder.bd import BrownDwarf

# class for the fields (potentially either many or one)
class Fields():
    def __init__(self, file=None, ra = None, dec = None, bd = None, n_arcmin=5):
        self.n_arcmin = n_arcmin
        
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
        dl.queryClient.getClient(profile='default', svc_url='https://datalab.noirlab.edu/query')
        
        if file == None:
            q = """SELECT
                        ls_id, ra, dec,  dered_mag_g, dered_mag_r, dered_mag_w1, dered_mag_w2, dered_mag_w3, dered_mag_w4, dered_mag_z, gaia_duplicated_source, pmdec, pmra, psfsize_g, psfsize_r, psfsize_z, ref_cat, ref_epoch, ref_id, type
                    FROM
                        ls_dr8.tractor
                    WHERE
                        't' = Q3C_RADIAL_QUERY(ra, dec,  {} , {} ,  ({}/60)) """.format(float(self.ra), float(self.dec), float(self.n_arcmin))
            res = qc.query(sql=q)
            self.stars = convert(res,'pandas')
        
        else:
            self.file = file
            self.stars = pd.read_csv(self.file)
        
        self.stars = self.filter_stars_only(self.stars)
        self.stars = self.filter_stars_mag(self.stars)

        # create array of paths for each star
        self.star_paths = np.zeros(len(self.stars))

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
        df = df.dropna(subset=['dered_mag_g']) #drop nan

        df['dered_mag_g'] = pd.to_numeric(df['dered_mag_g'])

        df = df[df.dered_mag_g >= 0]
        df= df[df.dered_mag_g <= 23.95]

        return df
