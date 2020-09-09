
# create brown dwarf class
class BrownDwarf():
  def __init__(self, dataset):
    self.dataset = dataset
    
    # find the path of the brown dwarf over the years
    self.path = find_path()
    
    
  ##
  # name: path_list
  #
  # inputs: data from the brown dwarf (like initial position, parallax, etc) and ephemerides from JPL Horizons
  # outputs: list of coordinates of the brown dwarf through time, ra and dec ends of brown dwarf (for plot resizing)
  #
  # purpose: function to grab the path of the brown dwarf with specified years
  def find_path(self, eph_dict, start, end):
    place_l = list()
    coord_dict = dict()
    
    #first need to pull general data on brown dwarf and convert to arcseconds
    a_0 = self.dataset[1] * 3600
    d_0 = self.dataset[2] * 3600
    
    pi_trig = float(number_only(self.dataset[3])) / 1000
    mu_a = float(number_only(self.dataset[4])) / 1000
    mu_d = float(number_only(self.dataset[5])) / 1000
    
    #errors to add in if needed    
    pi_trig_e = float(self.dataset[3][len(self.dataset[3])-4:]) / 1000
    mu_a_e = float(self.dataset[4][len(self.dataset[4])-4:]) / 1000
    mu_d_e = float(self.dataset[5][len(self.dataset[5])-4:]) / 1000
    
    if error == 'high':
        pi_trig += pi_trig_e
        mu_a += mu_a_e
        mu_d += mu_d_e
    elif error == 'low':
        pi_trig -= pi_trig_e
        mu_a -= mu_a_e
        mu_d -= mu_d_e
        
    t_0 = start #when observations happened
    
    #run through each ephemeride coordinate/time (time as months)
    for coord in eph_dict:
        #converting coord to year
        t = Time(float(coord), format='jd')
        t.format = 'jyear'
        t = t.value

        #cue formula for ra and dec at a given time.
        
        d_prime = d_0 + (mu_d * (t - t_0))
        #converting d to rad
        d_prime_r = float(d_prime / 206265)

        a_prime = a_0 + (mu_a * (t - t_0) / (np.cos(d_prime_r)))
        #convert a to rad
        a_prime_r = float(a_prime / 206265)

        a_t = a_prime + ((pi_trig * ((eph_dict[coord][0] * np.sin(a_prime_r)) - (eph_dict[coord][1] * np.cos(a_prime_r))) / np.cos(d_prime_r)))
        d_t = d_prime + (pi_trig * ((eph_dict[coord][0] * np.cos(a_prime_r) * np.sin(d_prime_r)) + (eph_dict[coord][1] * np.sin(a_prime_r) * np.sin(d_prime_r)) - (eph_dict[coord][2] * np.cos(d_prime_r))))
        
        # make delta a_t and d_t
        a_t -= a_0
        d_t -= d_0
        
        #convert a_t and d_t to degrees
        a_t = a_t / 3600
        d_t = d_t / 3600
        
        #add to the coord_dict with format: coord_dict[time] = [RA, Dec]
        coord_dict[t] = [a_t, d_t]
    
    #find list of alpha and dec to find the end points (for graphing purposes: to zoom into the path itself)
    a_list = list()
    d_list = list()
    for i in coord_dict:
        a_list.append(coord_dict[i][0])
        d_list.append(coord_dict[i][1])
    plt.scatter(a_list, d_list, s=0.5)

    a_ends = [list(coord_dict.values())[0][0], list(coord_dict.values())[len(a_list) - 1][0]]
    d_ends = [list(coord_dict.values())[0][1], list(coord_dict.values())[len(a_list) - 1][1]]
    
    #find mag_mu (for the title on the graph)
    mag_mu = math.sqrt((math.pow(mu_a, 2) + math.pow(mu_d, 2)))

    return coord_dict, a_ends, d_ends, mag_mu
