#####
# This file runs the MC algorithm for the slope 2d test case
#####

import numpy as np
import monte_carlo_fn as mc_fn
import build_fns_log as bd_fns
import time
from scipy import interpolate
from bathy import Bathymetry

# function to generate random sample
def sample():
    np.random.seed()
    return (np.random.uniform(0.035, 0.5, 1))[0]

# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_xbeach_2d
no_parallel_processes = 48 # should be equal to number of cores

# directory where XBeach files will be dumped and code will run - this needs to be changed to reflect name of folder
path_stem = '/rds/general/user/mc4117/home/MLMC_Code/slope_2d_mc/1/'

variable_name = 'bed_slope_2d' # name of uncertain variable
M = 2 # integer factor that meshsize is refined by at each level

t1 = time.time()

L_eps = 9 # fineness of grid
    
bathy = Bathymetry()

bathy.length_x = 1250
bathy.length_y = 1000
bathy.dx = bathy.length_x/M**L_eps
bathy.dy = bathy.length_y/M**L_eps

bathy.mc_slope_test(bathy.dx, bathy.dy, 0.6, 0.15, 0.15)

# interpolation function used to generate bed on new x and y grid
interp_orig_fn = interpolate.interp2d(bathy.x, bathy.y, bathy.z)

# run monte carlo algorithm
outputf, real_soln, file_name = mc_fn.monte_carlo_main(1000, M**L_eps, build_output, path_stem, sample, variable_name, no_parallel_processes, interp_orig_fn)

t2 = time.time()

# record output
logfile_output = open(file_name, "a+")

mc_fn.write(logfile_output, str(t2-t1))

logfile_output.close()
