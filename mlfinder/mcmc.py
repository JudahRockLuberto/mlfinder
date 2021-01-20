# basic imports
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import corner

from astropy.io import ascii
from astropy.table import Table

from PyAstronomy import pyasl


# class imports
from mlfinder.bd import BrownDwarf
from mlfinder.fields import Fields
from mlfinder.events import FindEvents

class MonteCarlo():
    def __init__(self, bd, vary, events, which=0, samples=1000):
        # check if bd and fields are classes
        if not isinstance(bd, BrownDwarf):
            raise Exception('Brown dwarf must be an instance of the BrownDwarf() class.')

        if not isinstance(events, FindEvents):
            raise Exception('Events must an instance of the FindEvents() class.')
            
        # basic creation of class
        self.bd = bd
        self.event = events.event_table.iloc[which]
        
        self.samples = samples
        self.which = which
        
        self.vary = vary
    
    ##
    # Name: delta_ml_calc
    #
    # inputs: brown dwarf data, a theta
    # outputs: a delta_ml by Cushing's formula
    #
    # purpose: when I have called this function, I have the smallest theta between dwarfs and background stars for each
    #          dwarf. So I need to find the delta_ml to see how helpful microlensing would be.
    #
    def delta_ml_calc(self, theta):
        #get parallax and astrometric precision
        parallax = float(self.bd.pi) / 1000
        
        astro_precision = 0.2 #mas, cushing's example
        k = 8.144 #mas/solar masses

        #using helpful formula
        delta_ml = (theta * astro_precision) / (k * parallax)

        delta_ml = delta_ml / (9.548 * math.pow(10, -4)) #making into jupiter masses

        return delta_ml
    
    ##
    # name: sampler
    #
    # inputs: brown dwarf data, event data, number of samples
    # outputs: a list of mass uncertainties
    #
    # purpose: see probability of good measurements of brown dwarfs with markov chain monte carlo
    def sampler(self):
        # get "new" data for brown dwarf
        
        # grab random samples of whichever is in self.vary
        vary_data = {'ra': [self.bd.ra, self.bd.pm_ra],
                     'dec': [self.bd.dec, self.bd.pm_dec],
                     'pi': [self.bd.pi, self.bd.pm_pi],
                     'mu_alpha': [self.bd.mu_a, self.bd.pm_mu_a],
                     'mu_delta': [self.bd.mu_d, self.bd.pm_mu_d]}
        
        # add data to a list for ease
        all_data = [np.random.normal(loc=vary_data[i][0], scale=vary_data[i][1], size=self.samples) if i in self.vary else vary_data[i][0] for i in vary_data]

        # find rang to run through
        placement = {'ra':0, 'dec':1, 'pi':2, 'mu_alpha':3, 'mu_delta':4}
        
        vary_index = placement[self.vary[0]]
        length = len(all_data[vary_index])
        
        # run through each sample and get the measurement uncertainty
        mass_unc_list = list()
        for i in range(length):
            if i % 100 == 0:
                print(i)
                
            # grab data if needs to be indexed or not
            instance_data = [j[i] if isinstance(j, np.ndarray) else j for j in all_data]
            
            # create a BrownDwarf instance
            bd_new = BrownDwarf(np.array([instance_data[0], instance_data[1], instance_data[2], instance_data[3], instance_data[4]]), array_col_names=['ra', 'dec', 'pi', 'mu_alpha', 'mu_delta'])
            bd_path = bd_new.find_path(start=self.bd.start, end=self.bd.end, step=self.bd.step)
            
            # take star info from event table and the brown dwarf path to find a list of distance
            separations = [pyasl.getAngDist(row.ra, row.dec, self.event.ra, self.event.dec) * 3600 for index, row in bd_path.iterrows()]
            
            # get minimum distance and calculate the mass uncertainty
            min_separation = min(separations)
            delta_ml = self.delta_ml_calc(min_separation)
            mass_unc_list.append(delta_ml)
        
        self.mass_unc_list = mass_unc_list
        return mass_unc_list