# basic imports
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import numpy as np
import pandas as pd

import astropy
from astropy.table import Table

from scipy.interpolate import interp1d

from PyAstronomy import pyasl

# class imports
from mlfinder.bd import BrownDwarf
from mlfinder.fields import Fields
from mlfinder.mcmc import MonteCarlo

# class to check for events
class FindEvents():
    def __init__(self, bd, fields, precision):
        # check if bd and fields are classes
        if not isinstance(bd, BrownDwarf):
            raise Exception('Brown dwarf must be an instance of the BrownDwarf() class.')
            
        if not isinstance(fields, Fields):
            raise Exception('Fields must an instance of the Fields() class.')
            
        # basic creation of class
        self.bd = bd
        self.stars = fields.stars
        
        self.m_jup_prec = precision
        
        self.coord_df = bd.coord_df
        
        # some helpful values
        self.theta_max = self.theta_max_calc()
        self.events_per_year = self.events_per_year_calc()
        
        # finding events
        self.event_table = self.find_events()
        
    ##
    # name: find_events
    #
    # inputs: bd and fields
    # outputs: table of closest approaches
    #And now for the event juicier stuff and the thing that calls upon the functions above.
    def find_events(self):
        # path
        coord_df = self.bd.coord_df
        
        # ends of path
        self.a_ends = [coord_df.ra[0], coord_df.ra[len(coord_df.ra) - 1]]
        self.d_ends = [coord_df.dec[0], coord_df.dec[len(coord_df.dec) - 1]]
        
        # add to df holding all closest dwarf and background star pairs to call on later.
        close_df = self.close_stars(a_ends = self.a_ends, d_ends = self.d_ends)

        return close_df
    
    ##
    # Name: theta_max_calc
    # 
    # inputs: data from the stars
    # outputs: theta_max for that individual star
    #
    # purpose: calculate theta max for a given parallax for each brown dwarf. So I can use to include any dwarf-background star
    #          pairs that are under the theta max for astrometric microlensing. I didn't want to miss any.
    #
    def theta_max_calc(self):
        # get parallax and astrometric precision.
        parallax = float(self.bd.bd_cut['pi']) / 1000 # arcseconds

        astro_precision = 0.2 #cushings example

        #constants
        big_g = 4.3 * math.pow(10, -3) #pc * solar_mass^-1 * (km/s)^2
        c_squared = 9 * math.pow(10, 10) #(km/s)^2
        d_l = 1 / parallax #in parsecs
        delta_ml = self.m_jup_prec * 9.548 * math.pow(10, -4) # masses of jupiter 
        
        k = 8.144 # mas/solar masses

        #actual formula. Used k one because was easiest.
        theta_max = (k * delta_ml * parallax) / (astro_precision)

        return theta_max
    
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
        parallax = float(self.bd.bd_cut['pi']) / 1000
        
        astro_precision = 0.2 #mas, cushing's example
        k = 8.144 #mas/solar masses

        #using helpful formula
        delta_ml = (theta * astro_precision) / (k * parallax)

        delta_ml = delta_ml / (9.548 * math.pow(10, -4)) #making into jupiter masses

        return delta_ml
    
    ##
    # Name: events_per_year_calc
    #
    # inputs: brown dwarf data, background stars
    # outputs: number of events per year
    #
    # purpose: previously wanted to know how many events per year should be occuring. I used it as a check for my microlensing
    #          events output. I may use it again, so I kept it.
    #
    def events_per_year_calc(self):
        #calculate the number of expected microlensing events per year for a given brown dwarf
        k = 8.144 #mas/solar masses

        #get parallax and astrometric precision
        # convert parallax to arcseconds
        parallax = float(self.bd.bd_cut['pi']) / 1000
        
        astro_precision = 0.2 #cushing's example

        mu_a = float(self.bd.bd_cut['mu_alpha']) / 1000
        mu_d = float(self.bd.bd_cut['mu_delta']) / 1000
        mu = math.sqrt((mu_a ** 2) + (mu_d ** 2))

        #formula from Cushing et al. I have delta_ml and delta_ml2 for two different Ml's (cushing and ours). Subsequently,
        #I also have two events per year. I last used our delta_ml, so I return our number
        sigma = len(self.stars) / (np.pi * ((5 * 60) ** 2)) #the surface density of stars per arcsecond^2 (#stars / area of view with radius 5 degrees)

        delta_ml = self.m_jup_prec * 9.548 * math.pow(10, -4) #solar mass of jupiter

        number = 2 * k * parallax * sigma * (delta_ml / astro_precision)

        return number, sigma
    
    ##
    # Name: add_to_close
    #
    # inputs: dataframe to add to, separation, time of separataion, index of bs, ra of bs, dec of bs, mass uncertainty.
    # outputs: updated dataframe
    #
    # purpose: to add rows to self.close_dict. there are two cases where i want to add (smallest theta after going through all the stars
    #          and if theta < theta_min)
    def add_to_close(self, close_df, object_name, sep, delta_m, bd_ra, bd_dec, ls_id, bs_ra, bs_dec, mag, gaia_pointsource, time_of_min):
        # set up dictionary and add to df
        value_dict = {'object_name': object_name,
                      'sep': sep,
                      'delta_m': delta_m,
                      'bd_ra': bd_ra,
                      'bd_dec': bd_dec,
                      'ls_id': ls_id,
                      'bs_ra': bs_ra,
                      'bs_dec': bs_dec,
                      'mag': mag,
                      'gaia_pointsource': gaia_pointsource,
                      'time_of_min': time_of_min
                     }

        return close_df.append(value_dict, ignore_index=True)
       
    ##
    # Name: close_stars
    #
    # inputs: ends of brown dwarf (to cut the stars looked at for nearness to save time),
    # outputs: dictionary of dwarf and background star pairs that make the cut of closest and/or under theta_max
    #
    # purpose: to run close_star_find function and keep on adding to the radius until close_star_find can find a star with the 
    #          given radius. Basically just increase the radius unl not given back None.
    #
    def close_stars(self, a_ends, d_ends):
        #continually increase the radius until hit more stars. makes sure get closest star in each while still keeping processing low
        close_df = pd.DataFrame()
        
        radius = 0
        while len(close_df) == 0:
            radius += 0.1
            close_df = self.close_star_find(a_ends = a_ends, d_ends = d_ends, radius = radius)
            
            if radius >= 1: #semi-random stopping point
                break
        
        return close_df

    ##
    # Name: close_star_find
    #
    # inputs: background stars, path of brown dwarf, ends of brown dwarf (to cut the stars looked at for nearness to save time),
    #         data from brown dwarf for a delta_ml calc, a theta_max for that specific brown dwarf if delta_ml is 4, radius that
    #         I'll explain in the purpose.
    # outputs: dictionary of dwarf and background star pairs that make the cut of closest and/or under theta_max
    #
    # purpose: to find the closest stars in a certain radius. In the function close_star, I increase the radius to save time in 
    #          calculation. I want to get the closest star but also not take lots of calc time for it. I also want to catch fast
    #          moving stars' events. All have a different radius, so to objectify it, I have an increasing radius until I hit
    #          a star
    def close_star_find(self, a_ends, d_ends, radius):
        #make ra and dec bounds to look at a star
        a_low = a_ends[0] - (radius * (a_ends[1] - a_ends[0]))
        a_high = a_ends[1] + (radius * (a_ends[1] - a_ends[0]))
        d_low = d_ends[0] - (radius * (d_ends[1] - d_ends[0]))
        d_high = d_ends[1] + (radius * (d_ends[1] - d_ends[0]))
        
        # make initial close_df
        close_df = pd.DataFrame(columns=['object_name', 'sep', 'delta_m', 'bd_ra', 'bd_dec', 'ls_id', 'bs_ra', 'bs_dec', 'mag', 'gaia_pointsource', 'time_of_min'])
        
        #The process for this function was to find the theta_min for any background star within the RA and Dec range found
        #through the radius. And then I process by taking the smallest delta_ml and any delta_mls lower than 4.

        # run through each background star
        theta_min = np.inf
        for i in range(len(self.stars)): 
            #if the star is within the range I am looking at
            # do checks   
            a_check = (abs(a_high - list(self.stars.ra)[i]) + abs(list(self.stars.ra)[i] - a_low)) == abs(a_high - a_low)
            d_check = (abs(d_high - list(self.stars.dec)[i]) + abs(list(self.stars.dec)[i] - d_low)) == abs(d_high - d_low)
            
            if a_check and d_check:
                theta_temp_min = np.inf
                
                #run through each brown dwarf data point in path
                for index, row in self.coord_df.iterrows(): 
                    a_1 = row['ra'] # deg
                    d_1 = row['dec'] # deg
                    a_2 = list(self.stars.ra)[i] # deg
                    d_2 = list(self.stars.dec)[i] # deg
                    
                    theta = pyasl.getAngDist(a_1, d_1, a_2, d_2) #angular difference in degrees
                    theta *= 3600 #convert from degrees to arcseconds

                    # if point is closer to background star than any point I have seen so far, make theta_min and record
                    # which background star it is.
                    if theta < theta_min:
                        theta_min = theta
                        time_of_min = row['time']
                        
                        ls_id = list(self.stars.ls_id)[i]
                        mag = list(self.stars.dered_mag_r)[i]
                        gaia_pointsource = list(self.stars.gaia_pointsource)[i]
                        
                        bd_ra, bd_dec = a_1, d_1
                        bs_ra, bs_dec = a_2, d_2
                        
                    # do the same thing as above, but for each brown dwarf that passes the checks. afterwards, check individual
                    # theta_temp_min and see if below self.theta_max
                    if theta < theta_temp_min:
                        temp_theta_min = theta
                        time_of_temp_min = row['time']
                        
                        temp_ls_id = list(self.stars.ls_id)[i]
                        temp_mag = list(self.stars.dered_mag_r)[i]
                        temp_gaia_pointsource = list(self.stars.gaia_pointsource)[i]
                        
                        temp_bd_ra, temp_bd_dec = a_1, d_1
                        temp_bs_ra, temp_bs_dec = a_2, d_2
                        
                # to make sure I catch possible events smaller than theta_max
                if temp_theta_min < self.theta_max:
                    temp_delta_ml = self.delta_ml_calc(temp_theta_min)
                    
                    close_df = self.add_to_close(close_df, self.bd.bd.object_name, temp_theta_min, temp_delta_ml, temp_bd_ra, temp_bd_dec, temp_ls_id, temp_bs_ra, temp_bs_dec, temp_mag, temp_gaia_pointsource, time_of_temp_min)
                                                 
        # find delta_ml for the smallest thetas and add to dictionary.
        # but only do it if goes within the checks (sometimes doesn't)
        if theta_min != np.inf:                         
            delta_ml = self.delta_ml_calc(theta_min)

            close_df = self.add_to_close(close_df, self.bd.bd.object_name, theta_min, delta_ml, bd_ra, bd_dec, ls_id, bs_ra, bs_dec, mag, gaia_pointsource, time_of_min)

        # now to find smallest sep in df or if lower than theta_max
        # find indices real quick
        if len(close_df) != 0:
            indices = [i for i, x in enumerate(close_df.sep) if x < self.theta_max]
            """
            # get rows
            if len(indices) != 0:
                close_df = close_df.iloc[indices,:]

            else:
                index = list(close_df.sep).index(min(close_df.sep))

                close_df = close_df.loc[index]
            """
        
        return close_df
    
    ##
    # Name: plot_event_path
    #
    # inputs: zoom, years between marker on plot
    # outputs: figure of brown dwarf path over background stars
    #
    # purpose: to create a plot of the path of the brown dwarf overlayed with background stars for sanity checks
    #
    #
    def plot_event_path(self, zoom=0.2, years=1, figsize=(10,10), gaia_check=False, legend=True, point_size=10, font_size=10, label_size=20):
         # basic setup
        fig = plt.figure(figsize=figsize)

        fig.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0,
            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.09, 0.09, 0.90, 0.90])

        # set titles
        plt.xlabel(r'$ \Delta \alpha_{J2000, deg} $', fontsize=label_size)
        plt.ylabel(r'$ \Delta \delta_{J2000, deg}$', fontsize=label_size)
        plt.title(str(self.bd.bd.object_name), fontsize=label_size)

        # set limits +-zoom length. I had initially done based on change of ra and dec, but that scaled the plot weird
        path_length = pyasl.getAngDist(self.a_ends[0] / 3600, self.d_ends[0] / 3600, self.a_ends[1] / 3600, self.d_ends[1] / 3600) #angular difference in degrees
        path_length *= 3600 #convert from degrees to arcseconds
        
        # see if add or subtract (plus or minus 1 which will be mult to zoom * path_length)
        a_dir = (self.a_ends[1] - self.a_ends[0]) / abs(self.a_ends[1] - self.a_ends[0])
        d_dir = (self.d_ends[1] - self.d_ends[0]) / abs(self.d_ends[1] - self.d_ends[0])
        
        # add lims themselves
        ax1.axis('equal')
        
        ax1.set_xlim(self.a_ends[0] + (-1 * a_dir * zoom * path_length), self.a_ends[1] + (a_dir * zoom * path_length))
        ax1.set_ylim(self.d_ends[0] + (-1 * d_dir * zoom * path_length), self.d_ends[1] + (d_dir * zoom * path_length))
        
        #make list of alpha and dec every 10 years and plot them with text as visual markers. easy to see in plot and see
        #direction the dwarf goes.   
        """measure_dict = {'1day': 365,
                        '7days': 52,
                        '1month': 12,
                        '3months': 4,
                        '4months': 3,
                        '1year' : 1}
        
        measure_in_year = measure_dict[self.bd.step]

        a_years = [i for i in self.bd.coord_df.ra if list(self.bd.coord_df.ra).index(i) % (years * measure_in_year) == 0]
        d_years = [i for i in self.bd.coord_df.dec if list(self.bd.coord_df.dec).index(i) % (years * measure_in_year) == 0]
        
        # plot year labels
        ax1.scatter(a_years, d_years, s=point_size+1, c='blue', marker="D")"""

        # adding text to 10 yr plot
        # finding placement to put plots based on change of ra/dec (10x more off with dec than ra bc of length of time)
        start_year = int(self.bd.start.split('-')[0])
        end_year = int(self.bd.end.split('-')[0])
        
        years = range(start_year, end_year+1, years)
        
        #ax1.annotate(years[0], (a_years[0], d_years[0]), fontsize=font_size)
        #for i, txt in enumerate(years):
        #    ax1.annotate(txt, (a_years[i], d_years[i]), fontsize=font_size)
            
        # plot the background stars
        if gaia_check == True:
            gaia = self.stars.gaia_pointsource
            gaia_c = ['red' if x==1 else 'grey' for x in gaia]
            
            gaia_handle = mpatches.Patch(color='red', label='$\it{Gaia}$ point source')
            not_gaia_handle = mpatches.Patch(color='grey', label='Not a $\it{Gaia}$ point source')
            
            if legend == True:
                ax1.legend(handles=[gaia_handle, not_gaia_handle])
        
        else:
            gaia_c = 'grey'

        ax1.scatter(self.stars.ra, self.stars.dec, s = point_size, c = gaia_c)
        
        # plot the brown dwarf path 
        ax1.scatter(self.coord_df.ra, self.coord_df.dec, s=point_size, c='orange')
        
        # add zoomed axis
        axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6
        axins.scatter(self.stars.ra, self.stars.dec, s = point_size, c = gaia_c)

        
        self.event_plot = fig
        
        return fig

    ##
    # Name: interpolate_shift
    #
    # intputs: shift, time
    # outputs: shift interpolated, r (in place of time)
    #
    # purpose: to make the centroid shift plot easier to understand (too few points isn't much of an argument)
    #          by linearly interpolating the centroid shift to each day, not month
    #
    def interpolate_shift(self, shift, time):
        r = np.linspace(min(time), max(time), 365 * (max(time) - min(time)))

        interp_shift = interp1d(time,
                                shift,
                                bounds_error=False, fill_value=np.nan,
                                kind='slinear')

        interp_shift_l = interp_shift(r)
        
        return interp_shift_l, r
    
    ##
    # Name: einstein_radius
    #
    # inputs: mass certainty
    # outputs: einstein radius for a given mass
    #
    # purpose: use in centroid_shift to calculate it
    #
    def einstein_radius(self, mass):
        parallax = float(self.bd.pi) / 1000 #convert from mas to arcseconds      

        #basic constants
        big_g = 4.3 * math.pow(10, -3) #pc * solar_mass^-1 * (km/s)^2
        c_squared = 9 * math.pow(10, 10) #(km/s)^2

        #calculate eintstein radius in mas
        e_r = np.sqrt(4 * mass * (9.55 * (10 ** -4)) * big_g * (1 / c_squared) * parallax) * 206265 * 1000

        return e_r

    ##
    # Name: centroid_shift
    #
    # inputs: 
    # outputs: table of centroid shift over time
    #
    # purpose: to find the centroid shift ove time for microlensing feasibility
    #
    def centroid_shift(self, which, masses=None):
        # if mass = None, run through Mjup=5,10,20,40
        if masses == None:
            mjups = [5,10,20,40]
            
        else:
            mjups = list(masses)
            
        self.mjups = mjups
            
        # find respective einstein radii
        self.einstein_radii = [self.einstein_radius(mass) for mass in mjups]
            
        # calculate for all the masses and have their centroid shifts
        # create the dataframe
        shift_df = pd.DataFrame(columns=['time'] + ['shift_' + str(mass) for mass in mjups])
        mag_df = pd.DataFrame(columns=['time'] + ['mag_' + str(mass) for mass in mjups])
        theta_list = list()
        for index, row in self.bd.coord_df.iterrows():
            # create dict of a row for the df
            temp_dict_shift = {'time' : row.time}
            temp_dict_mag = {'time' : row.time}

            # loop through each mjup and add it to temp_dict for each mass
            for j in range(len(mjups)):
                theta = pyasl.getAngDist(row.ra, row.dec, list(self.event_table.bs_ra)[which] , list(self.event_table.bs_dec)[which])

                #convert from degrees to mas
                theta *=  3600 * 1000
                theta_norm = theta / self.einstein_radii[j]
                
                shift = ((self.einstein_radii[j]) * theta_norm) / ((theta_norm ** 2) + 2)
                mag = ((theta_norm ** 2) + 2) / (theta_norm * math.sqrt((theta_norm ** 2) + 4))
                
                temp_dict_shift['shift_{}'.format(str(mjups[j]))] = shift
                temp_dict_mag['mag_{}'.format(str(mjups[j]))] = mag
                theta_list.append(theta)
                
            # add each row to the df
            shift_df = shift_df.append(temp_dict_shift, ignore_index=True)
            mag_df = mag_df.append(temp_dict_mag, ignore_index=True)    
            
        self.theta_list = theta_list
        self.shift_df = shift_df
        self.mag_df = mag_df

        return shift_df
        
    ##
    # Name: plot_shift
    #
    # inputs: shifts over time (arbitrary number)
    # outputs: plot of all the shifts over time
    #
    # purpose: visually see centroid shift
    #
    def plot_shift(self, figsize=(10,10)):    
        # plot each mass' centroid shift
        
        # basic setup
        fig = plt.figure(figsize=figsize)

        fig.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0,
            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.09, 0.09, 0.90, 0.90])

        # set titles
        ax1.set_xlabel(r'Time (yrs)', fontsize=20)
        ax1.set_ylabel(r'$ \delta_{m}(t)$ (mas) ', fontsize=20)

        ax1.tick_params(axis='both', labelsize=16)
        
        # lines for centroid shift plot
        """
        ax1.axvline(2021.7479)
        ax1.axvline(2022.7452)
        
        ax1.axvspan(2020.5325119780969, 2020.6830937713892, color='red', alpha=0.3)
        ax1.axvspan(2021.5345653661875, 2021.68514715948, color='red', alpha=0.3)
        """

        # add an arbitrary number of shifts after interpolating
        colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']

        for i in range(len(self.shift_df.columns) - 1):
            name = self.shift_df.columns[i + 1]
            number = name.split('_')[1]
            
            # interpolate the shift
            interp_shift, interp_time = self.interpolate_shift(self.shift_df[name], self.shift_df['time'])
            
            shift = ax1.scatter(interp_time, interp_shift, c=colors[i], s=6, label = number + r' M$_\mathrm{jup}$')
            
        # find where to set xlim based on peak: find point closest to half_max, get its index,
        # find the difference of max time to this time
        shift_col = list(self.shift_df[self.shift_df.columns[1]])
        time_col = list(self.shift_df[self.shift_df.columns[0]])
        
        half_max = max(shift_col) / 2

        closest = np.array(shift_col).flat[np.abs(np.array(shift_col) - half_max).argmin()]
        closest_index = shift_col.index(closest)
        
        max_index = shift_col.index(max(shift_col))
        
        time_dif = abs(time_col[max_index] - time_col[closest_index])
        
        # make axis to 2 * time difference
        ax1.set_xlim(time_col[max_index] - (2 * time_dif), time_col[max_index] + (2 * time_dif))

        # set tick lables and the like
        ax1.ticklabel_format(useOffset=False)

        ax1.legend(fontsize=20, markerscale=7)

        self.shift_fig = fig

        return fig

    ##
    # Name: plot_mag
    #
    # inputs: mags over time (arbitrary number)
    # outputs: plot of all the mags over time
    #
    # purpose: visually see centroid shift
    #
    def plot_mag(self, figsize=(10,10)):        
        # plot each mass' centroid shift

        # basic setup
        fig = plt.figure(figsize=figsize)

        fig.subplots_adjust(
            left=0.0, right=1.0, bottom=0.0, top=1.0,
            wspace=0.00, hspace=0.00)

        ax1 = fig.add_axes([0.09, 0.09, 0.90, 0.90])

        # set titles
        ax1.set_xlabel(r'Time (yrs)', fontsize=20)
        ax1.set_ylabel(r'$ \delta_{c}(t) - 1$ (mas) ', fontsize=20)

        ax1.tick_params(axis='both', labelsize=16)

        # add an arbitrary number of shifts after interpolating
        for i in range(len(self.mag_df.columns) - 1):
            name = self.mag_df.columns[i + 1]
            number = name.split('_')[1]
            
            # interpolate the shift
            interp_shift, interp_time = self.interpolate_shift(self.mag_df[name], self.mag_df['time'])
            
            # reduce by 1
            interp_shift_reduced = np.array(interp_shift) - 1
            
            shift = ax1.scatter(interp_time, interp_shift_reduced, s=2, label = number + r' M$_\mathrm{jup}$')
            
        # find where to set xlim based on peak: find point closest to half_max, get its index,
        # find the difference of max time to this time
        mag_col = list((self.mag_df[self.mag_df.columns[1]]))
        mag_col_reduced = list((self.mag_df[self.mag_df.columns[1]]) - 1)
        time_col = list(self.mag_df[self.mag_df.columns[0]])
        
        half_max = max(mag_col_reduced) / 2
    
        closest = np.array(mag_col_reduced).flat[np.abs(np.array(mag_col_reduced) - half_max).argmin()] + 1
        closest_index = mag_col.index(closest)
        
        max_index = mag_col.index(max(mag_col))
        
        time_dif = abs(time_col[max_index] - time_col[closest_index])
        
        # make axis to 2 * time difference
        ax1.set_xlim(time_col[max_index] - (2 * time_dif), time_col[max_index] + (2 * time_dif))
        
        # add a lower y limit (because plot starts at y=1)
        #ax1.set_ylim(bottom=0.99)

        # set tick lables and the like
        ax1.ticklabel_format(useOffset=False)

        ax1.legend(fontsize=20, markerscale=7)

        self.shift_fig = fig

        return fig

    ##
    # Name: event_mcmc
    #
    # inputs: which possible event, charteristics varying, number of samples
    # outputs: array of mass uncertainties
    #
    # purpose: find the possibility of events
    #
    def event_mcmc(self, vary, which=0, samples=1000):
        # create instance
        mcmc = MonteCarlo(self.bd, vary, self.event_table, which, samples)
        
        # find mass uncertainties
        uncertainties = mcmc.sampler()
        
        return uncertainties
