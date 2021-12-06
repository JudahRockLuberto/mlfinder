# basic imports
import time
from time import strptime

import numpy as np
import pandas as pd

import astropy
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astroquery.gaia import Gaia

from astroquery.jplhorizons import Horizons

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
            
            self.bd = bd
            
        else:
            raise Exception('Brown Dwarf data needs to either be ra/dec, an astropy table or pandas table of the dwarf data, or the brown dwarf class.')
        
        # now to grab the star info
        dl.queryClient.getClient(profile='default', svc_url='https://datalab.noirlab.edu/query')
        
        if file == None:
            q = """SELECT
                        ls_id, ra, dec,  dered_mag_g, dered_mag_r, dered_mag_w1, dered_mag_w2, dered_mag_w3, dered_mag_w4, dered_mag_z, gaia_phot_g_mean_mag, gaia_duplicated_source, pmdec, pmra, psfsize_g, psfsize_r, psfsize_z, ref_cat, ref_epoch, ref_id, type
                    FROM
                        ls_dr9.tractor
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

    ##
    # Name: get_gaia_data
    #
    # inputs: index of star
    # outputs: pandas table of position and motion (and errors) of star in gaia
    #
    # purpose: if the star isn't in gaia, it is far enough away to not have motion,
    #          but this will find if the background star is in gaia (and need to calc motion)
    def get_gaia_data(self, index):
        # get data about star
        row = self.stars.iloc[index]
        ls_id = row.ls_id

        # get gaia id of star from decals
        q = """SELECT
                ra1, dec1, id1, ra2, dec2, id2, distance
            FROM
                ls_dr9.x1p5__tractor__gaia_edr3__gaia_source
            WHERE 
                id1 = {} """.format(ls_id)

        res = qc.query(sql=q)
        decals_data = convert(res,'pandas')

        # get gaia id (or input id = 0 if no gaia data)
        gaia_id = 0 if len(decals_data.id2) == 0 else decals_data.id2
            
        # use gaia id to get info about it
        query = """SELECT 
                    source_id, ra, ra_error, dec, dec_error, parallax, pmra, pmra_error, pmdec, pmdec_error
                FROM 
                    gaiadr2.gaia_source
                WHERE 
                    source_id = {} """.format(int(gaia_id))
        print(gaia_id)
        gaia_data = Gaia.launch_job(query).get_results().to_pandas()

        return gaia_data
    
    ##
    # Name: find_star_path
    #
    # inputs: index of star, mu_a and mu_d in mas, start time, end time, time in between each data point,
    # outputs: path of that star
    #
    # purpose: find the path of a particular star
    def find_star_path(self, index, pi_trig, mu_a, mu_d, start, end, step='1month'):
        # creating an empty pandas dataframe bc easiest to work with
        coord_df = pd.DataFrame(columns=['time', 'ra', 'dec'])

        #first need to pull general data on star and convert to arcseconds
        a_0 = list(self.stars.ra)[index] * 3600
        d_0 = list(self.stars.dec)[index] * 3600

        mu_a = mu_a / 1000
        mu_d = mu_d / 1000

        # make inputted times into jd -- note that t_split is temp, so i reuse for the observed date and the start date

        # initial time
        if type(self.bd.observ_date) == float:
            t_0 = self.bd.observ_date
        else:
            t_split = self.bd.observ_date.split('-')
            t_0 = float(t_split[0]) + (strptime(t_split[1],'%b').tm_mon / 12) + (float(t_split[2]) / 365) #when observations happened

        # start time
        if type(start) == float:
            t_start = start
        else:
            t_split = start.split('-')
            t_start = float(t_split[0]) + (strptime(t_split[1],'%b').tm_mon / 12) + (float(t_split[2]) / 365) #when observations happened

        # grab ephemerides in vector form
        obj = Horizons(id='399', id_type='majorbody',
                       epochs={'start':self.bd.observ_date, 'stop':end,
                               'step':step})

        vectors = obj.vectors()
        vectors = vectors['targetname', 'datetime_jd', 'x', 'y', 'z']

        #run through each ephemeride coordinate/time (time as months)
        for coord in vectors:
            #converting coord to year
            t = Time(float(coord[1]), format='jd')
            t.format = 'jyear'
            t = t.value

            #cue formula for ra and dec at a given time.
            d_prime = d_0 + (mu_d * (t - t_0))

            #converting d to rad
            d_prime_r = float(d_prime / 206265)

            a_prime = a_0 + (mu_a * (t - t_0) / (np.cos(d_prime_r)))

            #convert a to rad
            a_prime_r = float(a_prime / 206265)

            # actual equations
            a_t = a_prime + ((pi_trig * ((coord[2] * np.sin(a_prime_r)) - (coord[3] * np.cos(a_prime_r))) / np.cos(d_prime_r)))
            d_t = d_prime + (pi_trig * ((coord[2] * np.cos(a_prime_r) * np.sin(d_prime_r)) + (coord[3] * np.sin(a_prime_r) * np.sin(d_prime_r)) - (coord[4] * np.cos(d_prime_r))))

            #convert a_t and d_t to degrees
            a_t = a_t / 3600
            d_t = d_t / 3600

            #add to the coord dataframe,  but only if during or after when we want the start
            if t > t_start:
                coord_df = coord_df.append({'time': t, 'ra': a_t, 'dec': d_t}, ignore_index=True)

        return coord_df
