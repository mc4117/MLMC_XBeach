# Functions for running Monte Carlo algorithm

import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import sys

import time
import datetime

def write(logfile, msg):
    """
    Write to both sys.stdout and to a logfile.
    """
    logfile = open(logfile.name, 'a+')
    logfile.write(msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    logfile.close()

def multilevel_mc(path, qin, qout):
    """
    In this function the XBeach model is run
    """
    os.chdir(path)
    for (nf, build_output, sigma_function, variable_name, interp_fn_orig) in iter(qin.get, 'stop'):
        sigma = sigma_function()

        outputf = build_output(path, nf, sigma, variable_name, interp_fn_orig, nf)

        qout.put(outputf)


def _parallel_mc_mc(processes, path_stem, calc_formula, nf, build_output, variable_name, sigma_function, interp_fn_orig, iteration):

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
            if (j+1)%1000 == 0:
                print(j)
            in_queue.put((nf, build_output, sigma_function, variable_name, interp_fn_orig))
        # send stop signals
        for i in range(processes):
            in_queue.put('stop')
        
        # collect output    
        results = []
        for i in range(iteration):
            if (i+1)%1000 == 0:
                print(i)
            results.append(out_queue.get())

        outputf = [f for f in results]
    
        return outputf



def monte_carlo_main(N0, nf, build_output, path_stem, sample, variable_name, processes, interp_fn_orig):
    
    """
    Runs Monte Carlo algorithm
    
    N0: number of samples
    nf: number of meshgrid points
    build_output: function which defines the XBeach inputs files, runs XBeach and outputs results
    path_stem: location where XBeach input and output files are dumped
    sample: function used to generate random number
    variable_name: name of uncertain parameter
    processes: number of parallel runs
    interp_fn_orig: interpolation function used to generate new beds for different x and y grids
    """

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    # output file of key stats
    filename = "monte_carlo_real_answer_bedslope_" + st + ".txt"
    logfile = open(filename, "w+")
    logfile.close()

    t0 = time.time()
    # run MC algorithm
    outputf = _parallel_mc_mc(processes, path_stem, multilevel_mc, nf, build_output, variable_name, sample, interp_fn_orig, N0)
    
    t1 = time.time()  

    # csv of all MC samples from this run
    csv_name = "average_output_" + st + ".csv"

    pd.DataFrame(outputf).to_csv(csv_name, index = False)
    
    logfile = open(filename, "a+")#, 0)

    write(logfile, "Number of samples: %0d " % N0)
    write(logfile, "Total time: %f " % (t1-t0))
    write(logfile, "Monte Carlo real value: %f " % (np.mean(outputf)))
    write(logfile, "Monte Carlo error: %f " % (np.sqrt(np.var(outputf)/len(outputf))))

    logfile.close()

    print('total time: ' + str(t1 - t0))
    print('expected value: ' + str(np.mean(outputf)))


    return np.mean(outputf), np.sqrt(np.var(outputf)/len(outputf)), filename
