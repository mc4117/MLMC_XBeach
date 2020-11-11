#####
# This file runs the MC algorithm for the morphological beach test case
#####

import numpy as np
import pandas as pd

import monte_carlo_fn as mc_fn
from scipy import interpolate
import time
import datetime
import build_fns_log as bd_fns

import re
import logging

# read in bed files 
bed_file = open('bed.dep').read()
tmpfile = re.sub(' +', ' ', bed_file)

bed_mc_file = open('bed_mc.dep', "w+")
bed_mc_file.write(tmpfile)
bed_mc_file.close()

bed_file = pd.read_csv("bed_mc.dep", sep=' ', header = None)

bed_file = bed_file.dropna(axis = 1)
bed_file.columns = [np.arange(0, bed_file.shape[1])]

# read in x and y grids

x_file = np.array(pd.read_csv('x.grd', sep='  ', header = None).iloc[0])

y_file = np.array(pd.read_csv('y.grd', sep='  ', header = None)[0])

if len(x_file) != bed_file.shape[1]:
    raise ValueError("Error reading in bed file")

# generate random sample
def sample():
    np.random.seed()
    return np.random.uniform(0.0, 3.5)

# set interpolate function used for interpolating bed on new x-y grids
interp_fn = interpolate.interp2d(x_file, y_file, bed_file)
angles_fn = None

x = np.linspace(interp_fn.x_min, interp_fn.x_max, 2**12)
y1 = np.linspace(0, 10, 3)
z_orig = interp_fn(x, y1)

# calculate total volume underneath the slope using the trapezium rule
# (considering the minimum value of z to be the lower vertical limit)

z_min = min([min(z_orig[i]) for i in range(len(z_orig))])
diff_list = [float(z_orig[0][i])-z_min for i in range(len(z_orig[0]))]

vol_change = diff_list[0] + diff_list[-1]

for i in range(1, len(diff_list)-1):
    vol_change += 2*diff_list[i]

init_vol = 0.5*(2**(-12))*vol_change*(interp_fn.x_max - interp_fn.x_min)


lmin = 6  # minimum refinement level
lmax = 11  # maximum refinement level

# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_morphological_changes
wavetype = 41
no_runs = 1
no_parallel_processes = 32 # equal to number of cores available

# path where dumps XBeach input and output files
path_stem = '/rds/general/user/mc4117/ephemeral/MLMC_Code/morphological_mc/1/'


variable_name = 'Hm0' # name of uncertain variable
dictionary_type = 'w' # dictionary where uncertain variable lives

t1 = time.time()

M = 2 # integer factor that meshsize is refined by at each level
L_eps = 12 # fineness of grid

# run monte carlo algorithm
outputf, real_soln, file_name = mc_fn.monte_carlo_main(32, M**L_eps, lmax, lmin, init_vol, build_output, path_stem, sample, variable_name, no_parallel_processes, interp_fn, angles_fn, wavetype)

t2 = time.time()
    
# store output
logfile_output = open(file_name, "a+")

mc_fn.write(logfile_output, str(t2-t1))

logfile_output.close()
