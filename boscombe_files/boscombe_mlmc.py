#####
# This file runs the MLMC algorithm for the Boscombe Beach test case
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
    In this function, the XBeach model is run on level l and level (l+1) using the same random number
    """
    os.chdir(path)
    for (l, M, build_output, sigma_function, variable_name, Lmin, Lmax, wavetype, interp_fn, angles_fn) in iter(qin.get, 'stop'):
        # generate random sample
        sigma = sigma_function()
        nf = M**(l+1)

        outputf = build_output(path, nf, sigma, variable_name, interp_fn, angles_fn,  M**Lmax, M**Lmin, instat_type = wavetype)
            
        nc = max((M**l), 1)
        outputc = build_output(path, nc, sigma, variable_name, interp_fn, angles_fn, M**Lmax, M**Lmin, instat_type = wavetype)
        
        qout.put((outputf, outputc))


def _parallel_mc(processes, path_stem, calc_formula, l, M, build_output, variable_name, sigma_function, interp_fn, angles_fn, Lmin, Lmax, wavetype, dictionary_type, iteration, normalisation_factor):
    """
    Split the tasks so the algorithm be parallelised and then collect the parallel output
    """
    
    # putting runs in queue
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

# generate random sample
def sample():
    np.random.seed()
    return np.random.uniform(0, 3)

M = 2  # integer factor that meshsize is refined by at each level
Lmin = 4  # minimum refinement level
Lmax = 7  # maximum refinement level
L = 7

# read in original bed file and x an y grids
bed_file = pd.read_csv("bed.dep", sep='  ', header = None)

x_file = np.array(pd.read_csv('x.grd', sep='  ', header = None).iloc[0])

y_file = np.array(pd.read_csv('y.grd', sep='  ', header = None)[0])

angle_dict = {}

# transform information from files into interpolation function so that bed information can easily be generated
# for different x and y grid spacings
interp_fn = interpolate.interp2d(x_file, y_file, bed_file, kind='cubic')

# smooth out bed using convolutions

for l in range(Lmin+1, L + 2):
    
    nf = 2**l
    l_max = 2**Lmax
    
    x = np.linspace(interp_fn.x_min, interp_fn.x_max, int(l_max))
    y = np.linspace(interp_fn.y_min, interp_fn.y_max, int(l_max))

    z = interp_fn(x, y)        

    print(l_max)
    print(nf)
    print(np.log2(l_max)-np.log2(nf))
    if np.log2(l_max) > np.log2(nf):
        k = np.ones((1, int(1+(np.log2(l_max)-np.log2(nf)))))
    elif np.log2(l_max) == np.log2(nf):
        k = np.ones((1, 1))
    else:
        k = np.ones((1, 1))

    k_list = k/sum(k[0])
        
    z_array = ndimage.convolve(z, k_list, mode='nearest')         

    interp_new_fn = interpolate.interp2d(x, y, z_array, kind='cubic')
    
    x_new = np.linspace(interp_new_fn.x_min, interp_new_fn.x_max, int(nf))
    y_new = np.linspace(interp_new_fn.y_min, interp_new_fn.y_max, int(nf))            
        
    znew = interp_new_fn(x_new, y_new) 
    
    total_angles = np.zeros(len(znew))
    
    # Calculate bedslope angle to be used in run-up formula.
    # This is done here to avoid having the calculate the angles at every call of the algorithm
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

    # create dictionary which stores angle values through an interpolation function for each y point on the grid
    angle_dict[nf] = angles_fn

# a list of all the levels that will be considered    
l_element = []
for i in range(Lmin, Lmax+1):
    l_element.append(i)

# epsilon values - for CDF calculation this should be set to epsilon = [0.002]
epsilon = [0.002, 0.004, 0.006, 0.008, 0.01]

N0 = 240  # number of samples used in preliminary run

# set the function which builds XBeach inputs, runs XBeach and records output
build_output = bd_fns.build_xbeach_boscombe_conv
no_runs = 1
no_parallel_processes = 40 # should be equal to number of cores


# directory where XBeach files will be dumped and code will run - this needs to be changed to reflect name of folder
path_stem = '/rds/general/user/mc4117/ephemeral/MLMC_Code/boscombe_conv_1_small/'

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

# set up unique file to store outputs
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

filename = "mlmc_xbeach_" + st + ".txt"
logfile = open(filename, "w+")
logfile.close()

# run MLMC algorithm
for i in range(no_runs):
    t0 = time.time()
    logfile = open(filename, "a+")#, 0)
    mlmc.write(logfile, str(no_parallel_processes))
    if init_test:
        alpha, beta, gamma, cost, var1, var2, del1 = mlmc.mlmc_test(no_parallel_processes, path_stem, _parallel_mc, multilevel, 'jonstable', dictionary_type, M, L, N0, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_fn, angles_fn = angle_dict)
        vardifflist.append(var1)
        varsing.append(var2)
        costlist.append(cost)
        dellist.append(del1)
        expectedlist.append(sum(del1))
    if eps_test:
        # these values come from the results of init_test and thus must be replaced if a new init_test run is performed
        alpha = 0.995703
        beta = 1.838690
        gamma = 2.521456
        var1 = [2.4302e-03, 8.1007e-04, 8.8298e-05, 7.2694e-05]
        del1 = [0.36023, -0.17017, -0.0076827, -0.0047053]
        Ns, varlevel, Plist, P_seq_list, cost_per_epsilon = mlmc.eps_only(alpha, beta, gamma, var1, del1, no_parallel_processes, path_stem, _parallel_mc, multilevel, 'jonstable', dictionary_type, M, L, 2, epsilon, Lmin, Lmax, logfile, build_output, variable_name, sample, interp_fn = interp_fn, angles_fn = angle_dict)
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
