# basic imports
import time
from time import strptime

import numpy as np
import pandas as pd

import astropy
from astropy.table import Table
from astropy.io import ascii
from astropy.time import Time

from PyAstronomy import pyasl

from astroquery.jplhorizons import Horizons

##
# name: find_info
#
# inputs: brown dwarf astropy table with columns titled 'object name', ra', 'dec', 'mu_alpha', 'mu_delta', 'pi' 
#         somewhere inside the table
# outputs: df with only columns of needed inputs
#
# purpose: used in BrownDwarf() and Fields() to pull info of the brown dwarf
def find_info(df, are_uncertainties):
    # keep certain columns
    keep_col = ['ra', 'dec', 'mu_alpha', 'mu_delta', 'pi']
    
    if all(are_uncertainties):
        keep_col += ['pm_ra', 'pm_dec', 'pm_pi', 'pm_mu_alpha', 'pm_mu_delta']

    df = df[keep_col]
    
    return df
  
# create brown dwarf class
class BrownDwarf():
    def __init__(self, bd, observ_date, array_col_names=None):
        # grab column names before make bd an np.array() (bc takes away column names)
        if isinstance(bd, pd.DataFrame):
            column_names = bd.columns.values
            
        if isinstance(bd, Table):
            column_names = bd.columns
            
        if isinstance(bd, np.ndarray):
            column_names = array_col_names
            
        self.observ_date = observ_date
            
        # check really quick if uncertainties in column names. If so, see that --all-- the uncertainties are there. else, return with issues
        uncertainties = ['pm_ra', 'pm_dec', 'pm_pi', 'pm_mu_alpha', 'pm_mu_delta']
        are_uncertainties = [True if i in column_names else False for i in uncertainties]
        
        if all(are_uncertainties) is False and any(are_uncertainties) is True:
            print('You only have some of the uncertainty values. Either include them all, or include none. All needed for ensuring good probabilities.')
            return None
        
        # convert bd to pandas dataframe -- an initial np.array conversion should work
        # first reshape bd array to (1, len(bd)), so dataframe is horizontal
        bd_reshaped = np.array(bd).reshape(1, len(column_names))
        
        self.bd = pd.DataFrame(bd_reshaped)
        
        # make bd have columns
        self.bd.columns = column_names
        
        # get basic data for the class
        # first change df into what want
        self.bd_cut= find_info(self.bd, are_uncertainties)
        
        # basic info
        self.ra = float(self.bd.ra)
        self.dec = float(self.bd.dec)
        self.mu_a = float(self.bd.mu_alpha)
        self.mu_d = float(self.bd.mu_delta)
        self.pi = float(self.bd.pi)
        
        # also add in uncertainties if present
        if all(are_uncertainties):
            self.pm_ra = float(self.bd.pm_ra)
            self.pm_dec = float(self.bd.pm_dec)
            self.pm_pi = float(self.bd.pm_pi)
            self.pm_mu_a = float(self.bd.pm_mu_alpha)
            self.pm_mu_d = float(self.bd.pm_mu_delta)
    
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
        a_0 = self.ra * 3600
        d_0 = self.dec * 3600

        pi_trig = self.pi / 1000
        mu_a = self.mu_a / 1000
        mu_d = self.mu_d / 1000
        
        # make inputted times into jd -- note that t_split is temp, so i reuse for the observed date and the start date
        
        # initial time
        if type(self.observ_date) == float:
            t_0 = self.observ_date
        else:
            t_split = self.observ_date.split('-')
            t_0 = float(t_split[0]) + (strptime(t_split[1],'%b').tm_mon / 12) + (float(t_split[2]) / 365) #when observations happened

        # start time
        if type(start) == float:
            t_start = start
        else:
            t_split = start.split('-')
            t_start = float(t_split[0]) + (strptime(t_split[1],'%b').tm_mon / 12) + (float(t_split[2]) / 365) #when observations happened

        # grab ephemerides in vector form
        obj = Horizons(id='399', id_type='majorbody',
                       epochs={'start':self.observ_date, 'stop':end,
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

            #convert a_t and d_t to degrees
            a_t = a_t / 3600
            d_t = d_t / 3600

            #add to the coord dataframe,  but only if during or after when we want the start
            if t > t_start:
                coord_df = coord_df.append({'time': [t], 'ra': [a_t], 'dec': [d_t]}, ignore_index=True)

        # put to BrownDwarf too
        self.coord_df = coord_df
        
        # add step, start, end because used for plotting in FindEvents()
        self.step = step
        self.start = start
        self.end = end
        
        return coord_df
