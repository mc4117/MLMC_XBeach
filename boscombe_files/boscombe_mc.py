#####
# This file runs the MC algorithm for the Boscombe Beach test case
#####

import numpy as np
import pandas as pd
import mlmc_fns as mlmc
import build_fns_log as bd_fns
from scipy import interpolate
from scipy import ndimage
import multiprocessing as mp
import os
import time
import datetime


def find_sign(array):
    return np.where(np.diff(np.signbit(array)))[0][0]

def multilevel(path, qin, qout):
    """
    In this function the XBeach model is run
    """
    os.chdir(path)
    for (l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, interp_fn, angles_fn) in iter(qin.get, 'stop'):

        sigma = sigma_function()
        nf = M**(l+1)
        
        # because this is for the monte carlo simulation, we only run the XBeach model for the finest level
        outputf = build_output(path, nf, sigma, variable_name, interp_fn, angles_fn,  M**Lmax, M**Lmin, instat_type = wavetype)
            
        nc = max((M**l), 1)

        outputc = 0

        qout.put((outputf, outputc))



def _parallel_mc(processes, path_stem, calc_formula, l, M, build_output, variable_name, sigma_function, interp_fn, angles_fn, Lmin, Lmax, wavetype, dictionary_type, iteration, normalisation_factor):
   
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """
    
    # putting runs into queue
    
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    future_res = []
    for i in range(processes):
        path = path_stem + str(i) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
    
        future_res.append(mp.Process(target = calc_formula, args = (path, in_queue, out_queue)))
        future_res[-1].start()

    for j in range(iteration):
        in_queue.put((l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, interp_fn, angles_fn))
    # send stop signals
    for i in range(processes):
        in_queue.put('stop')
        
    # collect output    
    results = []
    for i in range(iteration):
        if (i+1)%1000 == 0:
            print(i)
            mlmc.write(logfile, "%f" % (i))
            mlmc.write(logfile, "\n")
        results.append(out_queue.get())

    outputf = [f[0] for f in results]
    outputc = [f[1] for f in results]
    outputfc = 0
    outputfc2 = 0
    outputfc3 = 0
    outputfc4 = 0
    outputfsum = 0
    outputf2sum = 0
    
    # record output to csv file
    pd.DataFrame(outputf).to_csv(filename[:-4] + '.csv')
    
    for i in range(len(outputf)):
        if np.isnan(outputf[i]):
            print("nan")
        elif np.isnan(outputc[i]):
            print("nan")
        else:
            outputfc += (outputf[i] - outputc[i])
            outputfc2 += ((outputf[i] - outputc[i])**2)
            outputfc3 += ((outputf[i] - outputc[i])**3)
            outputfc4 += ((outputf[i] - outputc[i])**4)
            outputfsum += outputf[i]
            outputf2sum += outputf[i]**2
    
    sums = np.array([outputfc, outputfc2, outputfc3, outputfc4, outputfsum, outputf2sum])
    return sums

# function to generate random sample
def sample():
    np.random.seed()
    return np.random.uniform(0, 3)

M = 2
Lmin = 7  # minimum refinement level
Lmax = 7  # maximum refinement level (these are the same because for MC algorithm therefore all runs done at same level)
L = 7

# read in original bed file and x an y grids
bed_file = pd.read_csv("bed.dep", sep='  ', header = None)

x_file = np.array(pd.read_csv('x.grd', sep='  ', header = None).iloc[0])

y_file = np.array(pd.read_csv('y.grd', sep='  ', header = None)[0])

angle_dict = {}

# transform information from files into interpolation function so that bed information can easily be generated
# for different x and y grid spacings
interp_fn = interpolate.interp2d(x_file, y_file, bed_file, kind='cubic')

# calculate angles of bed for use in the EuroTop runup formula
for l in range(Lmin+1, L + 2):
    
    nf = 2**l
    l_max = 2**Lmax
    
    x = np.linspace(interp_fn.x_min, interp_fn.x_max, int(l_max))
    y = np.linspace(interp_fn.y_min, interp_fn.y_max, int(l_max))

    z = interp_fn(x, y)        

    interp_new_fn = interpolate.interp2d(x, y, z, kind='cubic')
    

    x_new = np.linspace(interp_new_fn.x_min, interp_new_fn.x_max, int(nf))
    y_new = np.linspace(interp_new_fn.y_min, interp_new_fn.y_max, int(nf))            
        
    znew = interp_new_fn(x_new, y_new) 
    
    total_angles = np.zeros(len(znew))

    for j in range(1, 2):
        angles = np.zeros(len(znew))

        for i in range(len(znew)):
            index = find_sign(znew[i])
            g = interpolate.interp1d([znew[i][index], znew[i][index+1]], [x_new[index], x_new[index+1]])
            height = znew[i][index+ j]
            x1 = -g(0) + x_new[index+j]
            angles[i] = height/x1
        total_angles += angles

    avg_angles = total_angles/(2**(l-4))


    angles_fn = interpolate.interp1d(y_new, avg_angles)

    angle_dict[nf] = angles_fn

# number of samples (note this file must be run multiple times to generate enough samples for MC - 96 are done at a time because of computational limits)
N0 = 96
# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_xbeach_boscombe_conv
no_runs = 1
no_parallel_processes = 48 # should be equal to number of cores

# path where dumps XBeach input and output files
path_stem = '/rds/general/user/mc4117/ephemeral/MLMC_Code/boscombe_cdt_2_mc_full/1/'

variable_name = 'Hm0' # name of uncertain variable
dictionary_type = 'w' # dictionary where uncertain variable lives

costlist = []
dellist = []

# set output txt file
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

filename = "mc_xbeach_" + st + ".txt"
logfile = open(filename, "w+")
logfile.close()

t0 = time.time()
logfile = open(filename, "a+")#, 0)
mlmc.write(logfile, str(no_parallel_processes))


# run monte carlo algorithm
cost, del1 = mlmc.mlmc_test(no_parallel_processes, path_stem, _parallel_mc, multilevel, 'jonstable', dictionary_type, M, L, N0, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_fn, angles_fn = angle_dict)
costlist.append(cost)
dellist.append(del1)

# record cost and expectation from this one run
logfile = open(filename, "a+")#, 0)
print('total time: ' + str(t1 - t0))
mlmc.write(logfile, "total time: %f" % (t1-t0))
mlmc.write(logfile, "\n")
