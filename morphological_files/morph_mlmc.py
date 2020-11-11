#####
# This file runs the MLMC algorithm for the morphology test case
#####


import numpy as np
import pandas as pd

import mlmc_fns as mlmc
from scipy import interpolate
import multiprocessing as mp
import time
import datetime
import build_fns_log as bd_fns

import re
import os
import subprocess
from netCDF4 import Dataset
import logging

def multilevel(path, qin, qout):
    """
    In this function, the XBeach model is run on level l and level (l+1) using the same random number
    """    
    os.chdir(path)
    for (l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, interp_fn, angles_fn, normalisation_factor) in iter(qin.get, 'stop'):
        # generate random sample
        sigma = sigma_function()

        nf = (M**(l+1))

        outputf = build_output(path, nf, sigma, variable_name, interp_fn, angles_fn,  M**Lmax, M**Lmin, instat_type = wavetype, normalisation_factor = normalisation_factor)
        
        nc = max((M**l), 1)
        outputc = build_output(path, nc, sigma, variable_name, interp_fn, angles_fn, M**Lmax, M**Lmin, instat_type = 'reuse', normalisation_factor = normalisation_factor)
        
        qout.put((outputf, outputc))

def _parallel_mc(processes, path_stem, calc_formula, l, M, build_output, variable_name, sigma_function, interp_fn, angles_fn, Lmin, Lmax, wavetype, dictionary_type, iteration, normalisation_factor):
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """
    
    # put in queue
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
        in_queue.put((l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, interp_fn, angles_fn, normalisation_factor))
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

    # record output values in csv (necessary for doing inverse sampling)
    try:
        print('output' + str(l+1)+'.csv')
        outputprevf = pd.read_csv('output' + str(l+1)+'.csv')
        outputf_df = pd.DataFrame(outputf, columns = ['output'])
        pd.concat([outputf_df, outputprevf]).to_csv('output' + str(l+1)+'.csv', index = False)
        print(len(pd.concat([outputf_df, outputprevf])))
    except:
        pd.DataFrame(outputf, columns = ['output']).to_csv('output' + str(l+1) + '.csv', index = False)
        
    try:
        print('output' + str(l)+'.csv')        
        outputprevc = pd.read_csv('output' + str(l)+'.csv')
        outputc_df = pd.DataFrame(outputc, columns = ['output'])
        pd.concat([outputc_df, outputprevc]).to_csv('output' + str(l)+'.csv', index = False)
        print(len(pd.concat([outputc_df, outputprevc])))
    except:
        pd.DataFrame(outputc, columns = ['output']).to_csv('output' + str(l) + '.csv', index = False) 
    
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

M = 2 # integer factor that meshsize is refined by at each level
Lmin = 6  # minimum refinement level
Lmax = 11  # maximum refinement level
L = 10
l_element = []
for i in range(Lmin, Lmax+1):
    l_element.append(i)

# epsilon values - for CDF calculation this should be set to epsilon = [0.000015]
epsilon = [0.000015, 0.000025, 0.00005, 0.0001]
N0 = 750 # number of samples used in preliminary run

# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_morphological_changes
wavetype = 41
no_runs = 1
no_parallel_processes = 40 # should be equal to number of cores

# directory where XBeach files will be dumped and code will run - this needs to be changed to reflect name of folder
path_stem = '/rds/general/user/mc4117/home/MLMC_Code/morphological_ero2/'

variable_name = 'Hm0' # name of uncertain variable
dictionary_type = 'w' # dictionary where uncertain variable lives

# choose whether doing a preliminary run of MLMC (init_test = True) or full run (eps_list = True)
eps_test = True
init_test = False

Nslist = []
expectedlist = []
cumulP_list = []
cumulP_seq_list = []
cumul_timecost = []
vardifflist = []
varsing = []
costlist = []
dellist = []
varlist = []

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# set up a unique file to store outputs
filename = "mlmc_xbeach_" + st + ".txt"
logfile = open(filename, "w+")
logfile.close()

# run MLMC algorithm
for i in range(no_runs):
    t0 = time.time()
    logfile = open(filename, "a+")#, 0)
    mlmc.write(logfile, str(no_parallel_processes))
    if init_test == True:
        alpha_mod, beta, gamma, cost, var1, var2, del1 = mlmc.mlmc_test(no_parallel_processes, path_stem, _parallel_mc, multilevel, wavetype, dictionary_type, M, L, N0, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_fn, normalisation_factor = init_vol)
        vardifflist.append(var1)
        varsing.append(var2)
        costlist.append(cost)
        dellist.append(del1)
        expectedlist.append(sum(del1))
    if eps_test == True:
        # these values come from the results of init_test and thus must be replaced if a new init_test run is performed
        alpha_mod = 0.921053
        beta  = 1.970078  
        gamma = 2.400694
        var1 = [5.9081e-07, 6.3646e-08, 1.9078e-08, 1.0362e-08, 1.5862e-09]
        del1 = [1.0026e-03, 1.2740e-04, 1.0285e-4, 9.3601e-05, 4.8057e-05]
        Ns, varlevel, Plist, P_seq_list, cost_per_epsilon = mlmc.eps_only(alpha_mod, beta, gamma, var1, del1, no_parallel_processes, path_stem, _parallel_mc, multilevel, wavetype, dictionary_type, M, L, 2, epsilon, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_fn, angles_fn = 0, normalisation_factor = init_vol)
        Nslist.append(Ns)
        varlist.append(varlevel)
        cumulP_list.append(Plist)
        cumulP_seq_list.append(P_seq_list)
        cumul_timecost.append(cost_per_epsilon)
        t1 = time.time()
    else:
        t1 = time.time()
    logfile.close()

    
# The rest of this file records the outputs of the MLMC algorithm
logfile = open(filename, "a+")#, 0)
print('total time: ' + str(t1 - t0))
mlmc.write(logfile, "total time: %f" % (t1-t0))
mlmc.write(logfile, "\n")

if eps_test:
    print(varlevel)
    mlmc.write(logfile, "Varlevel:")
    mlmc.write(logfile, str(varlevel))
    print(Nslist)
    mlmc.write(logfile, "\n")
    mlmc.write(logfile, "Ns:")
    mlmc.write(logfile, str(Nslist))

    for j in range(len(varlist)):
        var_eps = [np.sqrt(varlist[j][0][i])/np.sqrt(Nslist[j][0][i]) for i in range(len(varlist[j][0]))]
    
        print('error bar: ' + str(sum(var_eps)))
    
        mlmc.write(logfile, "\n")
        mlmc.write(logfile, "Error bar:")
        mlmc.write(logfile, str(sum(var_eps)))
logfile.close()

if init_test:
    
    filename_orig_output = "mlmc_xbeach_" + st + "_orig_output_" + str(i) + ".txt"
    logfile_output = open(filename_orig_output, "w+")
    mlmc.write(logfile_output, str(no_parallel_processes))
    mlmc.write(logfile_output, "l_element:")
    mlmc.write(logfile_output, str(l_element))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "var_diff:")
    mlmc.write(logfile_output, str(vardifflist))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "var_sing:")
    mlmc.write(logfile_output, str(varsing))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "cost:")
    mlmc.write(logfile_output, str(costlist))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "del1:")
    mlmc.write(logfile_output, str(dellist))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "sum(del1):")
    mlmc.write(logfile_output, str([sum(delsum) for delsum in dellist]))
    mlmc.write(logfile_output, "\n")
    logfile_output.close()

if eps_test:
    filename_eps_output = "mlmc_xbeach_" + st + "_eps_output_" + str(i) + ".txt"
    logfile_output = open(filename_eps_output, "w+")
    mlmc.write(logfile_output, str(no_parallel_processes))
    mlmc.write(logfile_output, "l_element:")
    mlmc.write(logfile_output, str(l_element))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "epsilon:")
    mlmc.write(logfile_output, str(epsilon))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "Ns:")
    mlmc.write(logfile_output, str(Nslist))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "Plist:")
    mlmc.write(logfile_output, str(cumulP_list))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "P_seq_list:")
    mlmc.write(logfile_output, str(cumulP_seq_list))
    mlmc.write(logfile_output, "\n")
    mlmc.write(logfile_output, "cost_per_epsilon:")
    mlmc.write(logfile_output, str(cumul_timecost))
    logfile_output.close()
    
    print('expected value: ' + str(np.sum(cumulP_list)/len(cumulP_list[0])))
