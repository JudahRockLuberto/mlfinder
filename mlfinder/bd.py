# basic imports
import time

import numpy as np
import pandas as pd

import astropy
from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time

from PyAstronomy import pyasl

from astroquery.jplhorizons import Horizons

# misc. functions

##
# name: find_info
#
# inputs: brown dwarf astropy table with columns titled 'object name', ra', 'dec', 'mu_alpha', 'mu_delta', 'pi' 
#         somewhere inside the table
# outputs: df with only columns of needed inputs
#
# purpose: used in BrownDwarf() and Fields() to pull info of the brown dwarf
def find_info(df):
    # keep certain columns
    df = df[['ra', 'dec', 'pi', 'mu_alpha', 'mu_delta']]
    
    return df
  
# create brown dwarf class
class BrownDwarf():
    def __init__(self, bd):
        # grab column names before make bd an np.array() (bc takes away column names)
        if isinstance(bd, pd.DataFrame):
            column_names = bd.columns.values
            
        if isinstance(bd, Table):
            column_names = bd.columns
            
        if isinstance(bd, np.ndarray):
            column_names = ['ra', 'dec', 'pi', 'mu_alpha', 'mu_delta']
        
        # convert bd to pandas dataframe -- an initial np.array conversion should work
        self.bd = pd.DataFrame(np.array(bd))
        
        # make bd have columns
        self.bd.columns = column_names
        
        # get basic data for the class
        # first change df into what want
        self.bd_cut= find_info(self.bd)
        
        # basic info (made weird bc pandas is weird)
        self.ra = float(bd['ra'].values)
        self.dec = float(bd['dec'].values)
        self.mu_a = float(bd['mu_alpha'].values)
        self.mu_d = float(bd['mu_delta'].values)
        self.pi = float(bd['pi'].values)
    
    ##
    # name: path_list
    #
    # inputs: data from the brown dwarf (like initial position, parallax, etc) and ephemerides from JPL Horizons
    # outputs: list of coordinates of the brown dwarf through time, ra and dec ends of brown dwarf (for plot resizing)
    #
    # purpose: function to grab the path of the brown dwarf with specified years
    def find_path(self, start, end, step='1month'):
        # creating an empty pandas dataframe bc easiest to work with
        coord_df = pd.DataFrame(columns=['time', 'ra', 'dec'])

        #first need to pull general data on brown dwarf and convert to arcseconds
        a_0 = float(self.bd_cut['ra']) * 3600
        d_0 = float(self.bd_cut['dec']) * 3600

        pi_trig = float(self.bd_cut['pi']) / 1000
        mu_a = float(self.bd_cut['mu_alpha']) / 1000
        mu_d = float(self.bd_cut['mu_delta']) / 1000

        t_0 = float(start.split('-')[0]) #when observations happened
        
        # grab ephemerides in vector form
        obj = Horizons(id='399', id_type='majorbody',
                       epochs={'start':start, 'stop':end,
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

            a_t = a_prime + ((pi_trig * ((coord[2] * np.sin(a_prime_r)) - (coord[3] * np.cos(a_prime_r))) / np.cos(d_prime_r)))
            d_t = d_prime + (pi_trig * ((coord[2] * np.cos(a_prime_r) * np.sin(d_prime_r)) + (coord[3] * np.sin(a_prime_r) * np.sin(d_prime_r)) - (coord[4] * np.cos(d_prime_r))))

            # make delta a_t and d_t
            a_t -= a_0
            d_t -= d_0

            #convert a_t and d_t to degrees
            a_t = a_t / 3600
            d_t = d_t / 3600

            #add to the coord dataframe
            coord_df = coord_df.append({'time': t, 'ra': a_t, 'dec': d_t}, ignore_index=True)

        # put to BrownDwarf too
        self.coord_df = coord_df
        
        # add step, start, end because used for plotting in FindEvents()
        self.step = step
        self.start = start
        self.end = end
        
        return coord_df
