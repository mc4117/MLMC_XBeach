#####
# This file runs the MLMC algorithm for the slope 2d test case
#####

import numpy as np
import mlmc_fns as mlmc
import build_fns_log as bd_fns
import multiprocessing as mp
import os
import time
import datetime
from scipy import interpolate
from bathy import Bathymetry

import pandas as pd

def multilevel(path, qin, qout):
    """
    In this function, the XBeach model is run on level l and level (l+1) using the same random number
    """
    os.chdir(path)
    for (l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, dictionary_type, interp_fn) in iter(qin.get, 'stop'):
        
        # generate random sample
        sigma = sigma_function()
        nf = M**(l+1)
        outputf = build_output(path, nf, sigma, variable_name, interp_fn, M**Lmax, M**Lmin)

        nc = max(nf/M, 1)
        outputc = build_output(path, nc, sigma, variable_name, interp_fn, M**Lmax, M**Lmin)
        
        qout.put((outputf, outputc))

def _parallel_mc(processes, path_stem, calc_formula, l, M, build_output, variable_name, sigma_function, interp_fn, angles_fn, Lmin, Lmax, wavetype, dictionary_type, iteration):
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """   
    
    # putting runs into queues  
    in_queue = mp.Queue()
    out_queue = mp.Queue()
    future_res = []
    for i in range(processes):
        path = path_stem + str(i) + '/'
        if not os.path.exists(path):
	        os.makedirs(path)
    
        future_res.append(mp.Process(target = multilevel, args = (path, in_queue, out_queue)))
        future_res[-1].start()

    for j in range(iteration):
        in_queue.put((l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, dictionary_type, interp_fn))
    # send stop signals
    for i in range(processes):
        in_queue.put('stop')
        
    # collect output    
    results = []
    for i in range(iteration):
        if (i+1)%1000 == 0:
            print(i)
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
        outputfc += (outputf[i] - outputc[i])
        outputfc2 += ((outputf[i] - outputc[i])**2)
        outputfc3 += ((outputf[i] - outputc[i])**3)
        outputfc4 += ((outputf[i] - outputc[i])**4)
        outputfsum += outputf[i]
        outputf2sum += outputf[i]**2
    
    sums = np.array([outputfc, outputfc2, outputfc3, outputfc4, outputfsum, outputf2sum])
    return sums


# generate random sample
def sample():
    np.random.seed()
    return (np.random.uniform(0.035, 0.5, 1))[0]


M = 2 # integer factor that meshsize is refined by at each level
Lmin = 2  # minimum refinement level
Lmax = 8  # maximum refinement level
L = 7

# list of all levels considered
l_element = []
for i in range(Lmin, Lmax+1):
    l_element.append(i)

bathy = Bathymetry()

bathy.length_x = 1250
bathy.length_y = 1000
bathy.dx = bathy.length_x/M**Lmax
bathy.dy = bathy.length_y/M**Lmax
bathy.mc_slope_test(bathy.dx, bathy.dy, 0.6, 0.15, 0.15)

# interpolation function used to generate bed on new x and y grid
interp_orig_fn = interpolate.interp2d(bathy.x, bathy.y, bathy.z)

# epsilon values - for CDF calculation this should be set to epsilon = [0.001]
epsilon = [0.001, 0.0017, 0.0025, 0.0035, 0.005, 0.006]
N0 = 500 # number of samples used in preliminary run

# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_xbeach_2d
no_runs = 1
no_parallel_processes = 40 # should be equal to number of cores

# directory where XBeach files will be dumped and code will run - this needs to be changed to reflect name of folder
path_stem = '/rds/general/user/mc4117/home/MLMC_Code/slope_2d_interp_ag/'

variable_name = 'bed_slope_2d' # name of uncertain variable

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

# set up unique file to store outputs
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

filename = "mlmc_xbeach_" + st + ".txt"
logfile = open(filename, "w+")
logfile.close()

for i in range(no_runs):
    t0 = time.time()
    logfile = open(filename, "a+")#, 0)
    mlmc.write(logfile, str(no_parallel_processes))
    if init_test == True:
        alpha, beta, gamma, cost, var1, var2, del1 = mlmc.mlmc_test(no_parallel_processes, path_stem, _parallel_mc, multilevel, 'jons', 'w', M, L, N0, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_orig_fn)
        vardifflist.append(var1)
        varsing.append(var2)
        costlist.append(cost)
        dellist.append(del1)
        expectedlist.append(sum(del1))
    if eps_test == True:
        # these values come from the results of init_test and thus must be replaced if a new init_test run is performed          
        alpha = 1.177080
        beta = 1.690109
        gamma = 2.926862
        var1 = [0.0033921875000000323, 0.0006655625, 0.00029648046875, 9.73984375e-05, 2.5018310546874996e-05, 8.331298828125e-06]
        del1 = [0.64625, 0.01275, 0.0124375, 0.009125, 0.006171875, 0.003515625]
        Ns, varlevel, Plist, P_seq_list, cost_per_epsilon = mlmc.eps_only(alpha, beta, gamma, var1, del1, no_parallel_processes, path_stem, _parallel_mc, multilevel, 'jons', 'w', M, L, 2, epsilon, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_orig_fn, angles_fn = 0)
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
