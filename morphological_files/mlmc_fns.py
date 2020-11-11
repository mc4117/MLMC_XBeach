# Functions to run MLMC algorithm. This has been in part adapted from Patrick Farrell's code
# available at https://bitbucket.org/pefarrell/pymlmc/src/master/

import numpy as np
import timeit
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd


class WeakConvergenceFailure(Exception):
    pass

def write(logfile, msg):
    """
    Write to both sys.stdout and to a logfile.
    """
    logfile = open(logfile.name, 'a+')
    logfile.write(msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    logfile.close()

def mlmc_test(no_parallel_processes, path_stem, mlmc_fn, calc_formula, wavetype, dictionary_type, M, L, N0, Lmin, Lmax, logfile, build_output, variable_name, sigma_function, interp_fn = 0, angles_fn = 0, normalisation_factor = 1):
    """
    Multilevel Monte Carlo routine for preliminary run.

    no_parallel_processes: number of different parallel runs of the model
    path_stem: location where XBeach inputs and outputs will be dumped
    mlmc_fn: the user low-level routine. Its interface is
      sums = mlmc_fn(l, N)
    with inputs
      l = level
    and a np array of outputs
      sums[0] = sum(Pf-Pc)
      sums[1] = sum((Pf-Pc)**2)
      sums[2] = sum((Pf-Pc)**3)
      sums[3] = sum((Pf-Pc)**4)
      sums[4] = sum(Pf)
      sums[5] = sum(Pf**2)
      
    calc_formula: multilevel algorithm
    wavetype: type of wave used
    dictionary_type: dictionary where uncertain parameter lives
    M: refinement cost factor (2**gamma in general MLMC theorem)
    L: number of levels for convergence tests
    N0: initial number of samples for MLMC calculations of alpha and beta
    Lmin: minimum level
    Lmax: maximum level
    logfile: file where outputs are stored
    build_output: function which defines the XBeach inputs files, runs XBeach and outputs results
    variable_name: name of uncertain parameter
    sigma_function: function used to generate random number
    interp_fn: interpolation function used to generate new beds for different x and y grids [optional]
    angles_fn: interpolation function used to store bed angle for all values of y [optional]
    normalisation_factor: factor by which the output is divided by [optional]
    """

    # First, convergence tests


    del1 = []
    del2 = []
    var1 = []
    var2 = []
    chk1 = []
    kur1 = []
    cost = []

    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** Convergence tests, kurtosis, telescoping sum check ***\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)    cost")
    write(logfile, "    kurtosis     check \n-------------------------")
    write(logfile, "--------------------------------------------------\n")

    for l in range(Lmin, L+1):
        tic = timeit.default_timer()
        sums = mlmc_fn(processes = no_parallel_processes, path_stem = path_stem, calc_formula = calc_formula, l = l, M = M, build_output = build_output, variable_name = variable_name, sigma_function = sigma_function, interp_fn = interp_fn, angles_fn = angles_fn, Lmin = Lmin, Lmax = Lmax, wavetype = wavetype, dictionary_type = dictionary_type, iteration = N0, normalisation_factor = normalisation_factor)
        toc = timeit.default_timer()
        sums = sums/N0

        del1.append(sums[0])
        del2.append(sums[4])
        var1.append(sums[1]-sums[0]**2)
        var2.append(max(sums[5]-sums[4]**2, 1.0e-10)) # fix for cases with var = 0
        cost.append(toc - tic)
        
        # check that the (E[Pf - Pc]  + E[Pc] - E[Pf]) / 
        if l == Lmin:
            check = 0
        else:
            check =          abs(       del1[l-Lmin]  +      del2[l-1-Lmin]  -      del2[l-Lmin])
            #print("check not switched on")
            check = check / ( 3.0*(math.sqrt(var1[l-Lmin]) + math.sqrt(var2[l-1-Lmin]) + math.sqrt(var2[l-Lmin]) )/math.sqrt(N0))
        chk1.append(check)
        
        # calculate kurtosis
        if l == Lmin:
            kurt = 0.0
        else:
            kurt = (     sums[3]
                     - 4*sums[2]*sums[0]
                     + 6*sums[1]*sums[0]**2
                     - 3*sums[0]*sums[0]**3 ) / (sums[1]-sums[0]**2)**2
        kur1.append(kurt)
        write(logfile, "%2d   %8.4e  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e %8.4e \n" % \
                      (l, del1[l-Lmin], del2[l-Lmin], var1[l-Lmin], var2[l-Lmin], cost[l-Lmin], kur1[l-Lmin], chk1[l-Lmin]))


    if kur1[-1] > 100.0:
        write(logfile, "\n WARNING: kurtosis on finest level = %f \n" % kur1[-1]);
        write(logfile, " indicates MLMC correction dominated by a few rare paths; \n");
        write(logfile, " for information on the connection to variance of sample variances,\n");
        write(logfile, " see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n");


    if max(chk1) > 1.0:
        print("WARNING: maximum consistency error = %f indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied" % max(chk1))
        write(logfile, "\n WARNING: maximum consistency error = %f \n" % max(chk1))
        write(logfile, " indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied \n\n")

    write(logfile, "\n expected value after %d levels is %g \n" %(L, sum(del1)))

    # find the gradient of the line which goes through the points log_2 of E[Pf - Pc]
    first_non_zero_alpha = next((i for i, x in enumerate(del1) if x), None)
    print(first_non_zero_alpha)
    pa    = np.polyfit(range(first_non_zero_alpha+Lmin, L+1), np.log2(np.abs(del1[first_non_zero_alpha:len(del1)+1])), 1);  alpha = -pa[0];
    # find the gradient of the line which goes through the points log_2 of Var[Pf - Pc]
    first_non_zero_beta = next((i for i, x in enumerate(var1) if x), None)
    print(first_non_zero_beta)
    pb    = np.polyfit(range(first_non_zero_beta+Lmin, L+1), np.log2(np.abs(var1[first_non_zero_beta:len(var1)+1])), 1);  beta  = -pb[0];
    gamma = np.log2(cost[-1]/cost[-2]);

    write(logfile, "\n******************************************************\n");
    write(logfile, "*** Linear regression estimates of MLMC parameters ***\n");
    write(logfile, "******************************************************\n");
    write(logfile, "\n alpha = %f  (exponent for MLMC weak convergence)\n" % alpha);
    write(logfile, " beta  = %f  (exponent for MLMC variance) \n" % beta);
    write(logfile, " gamma = %f  (exponent for MLMC cost) \n" % gamma);
    
    
    return alpha, beta, gamma, cost, var1, var2, del1

def eps_only(alpha, beta, gamma, Vl_orig, ml_orig, no_parallel_processes, path_stem, mlmc_fn, calc_formula, wavetype, dictionary_type, M, L, min_num, Eps, Lmin, Lmax, logfile, build_output, variable_name, sigma_function, interp_fn = 0, angles_fn = 0, normalisation_factor = 1):


    """
    Multilevel Monte Carlo routine for full run.

    alpha, beta, gamma, Vl_orig, ml_orig: the first 5 inputs come from the output of the preliminary run.
        They are the convergence parameters, variance and mean.
    no_parallel_processes: number of different parallel runs of the model
    path_stem: location where XBeach inputs and outputs will be dumped
    mlmc_fn: the user low-level routine. Its interface is
      sums = mlmc_fn(l, N)
    with inputs
      l = level
    and a np array of outputs
      sums[0] = sum(Pf-Pc)
      sums[1] = sum((Pf-Pc)**2)
      sums[2] = sum((Pf-Pc)**3)
      sums[3] = sum((Pf-Pc)**4)
      sums[4] = sum(Pf)
      sums[5] = sum(Pf**2)
      
    calc_formula: multilevel algorithm
    wavetype: type of wave used
    dictionary_type: dictionary where uncertain parameter lives
    M: refinement cost factor (2**gamma in general MLMC theorem)
    L: number of levels for convergence tests
    min_num: minimum number of samples allowed at a level
    Eps: tolerance parameter to control accuracy
    Lmin: minimum level
    Lmax: maximum level
    logfile: file where outputs are stored
    build_output: function which defines the XBeach inputs files, runs XBeach and outputs results
    variable_name: name of uncertain parameter
    sigma_function: function used to generate random number
    interp_fn: interpolation function used to generate new beds for different x and y grids [optional]
    angles_fn: interpolation function used to store bed angle for all values of y [optional]
    normalisation_factor: factor by which the output is divided by [optional]
    """

    
    Nslisteps = []
    varlevel = []
    P_list = []
    P_seq_list = []
    cost_per_epsilon = []

    write(logfile, "\n");
    write(logfile, "***************************** \n");
    write(logfile, "*** MLMC complexity tests *** \n");
    write(logfile, "***************************** \n\n");
        
    for eps in Eps:
        first_pass = True
        tic = timeit.default_timer()
        write(logfile, "Epsilon: %.4f;" % (eps))
        write(logfile, "----------------------------------------------- \n");            

        if Lmax < Lmin:
            raise ValueError("Need Lmax >= Lmin")

        L = Lmin + 2

        Nl   = np.zeros(L+1-Lmin)
        suml = np.zeros((2, L+1-Lmin))
        dNl  = min_num*np.ones(L+1-Lmin)
        
        twobetal = [2**(-beta*i) for i in np.arange(Lmin, Lmin + len(Vl_orig))]
        
        c2list = []
        
        for i in range(len(twobetal)):
            c2list.append(Vl_orig[i]/twobetal[i])
        
        c2 = max(c2list)
        write(logfile, "c2: %.4f;" % (c2))

        while sum(dNl) > 0:

            # update sample sums
            if first_pass == False:
                write(logfile, "Epsilon: %.4f" % eps)
                write(logfile, " \n");
                write(logfile, 'dNl eps first pass: ')
                write(logfile, " ".join(["%9d" % n for n in dNl]))
                write(logfile, " \n");
                for l in range(0, L+1-Lmin):
                    if dNl[l] > 0:
                        # generate MLMC outputs
                        sums = mlmc_fn(no_parallel_processes, path_stem, calc_formula, l+Lmin, M, build_output, variable_name, sigma_function, interp_fn, angles_fn, Lmin, Lmax, wavetype = wavetype, dictionary_type = dictionary_type, iteration = int(dNl[l]), normalisation_factor = normalisation_factor)
                        kurt = (     sums[3]
                                - 4*sums[2]*sums[0]
                                + 6*sums[1]*sums[0]**2
                                - 3*sums[0]*sums[0]**3 ) / (sums[1]-sums[0]**2)**2
                            
                        if kurt > 100.0:
                            write(logfile, "\n WARNING: kurtosis on finest level = %f \n" % kurt)

                        Nl[l]      = Nl[l] + dNl[l]
                        suml[0, l] = suml[0, l] + sums[0]
                        suml[1, l] = suml[1, l] + sums[1]
                        write(logfile,'sum: ')
                        write(logfile, " ".join(["%.4f" % n for n in (suml[0, :])]))
                        write(logfile, " \n");
                        # compute absolute average and variance

                ml = np.abs(       suml[0, :]/Nl)
                Vl = np.maximum(0, suml[1, :]/Nl - ml**2)

            # set optimal number of additional samples
            if first_pass == True:
                Cl = 2**(gamma * np.arange(Lmin, L+1))
                # use formulas given in paper to calculate estimates for optimum number of samples
                if beta > gamma:
                    Ns = [np.ceil(2*(eps**(-2))*c2*((1-(2**(-(beta-gamma)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]
                else:
                    Ns = [np.ceil(2*(eps**(-2))*c2* (2**((-beta+gamma)*L/2))*  ((1-(2**(-(gamma-beta)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]
                write(logfile, " ".join(["%9d" % n for n in Ns]))
                first_pass = False
            else:
                Cl = 2**(gamma * np.arange(Lmin, L+1))
                if beta > gamma:
                    Ns = [np.ceil(2*(eps**(-2))*c2*((1-(2**(-(beta-gamma)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]
                else:
                    Ns = [np.ceil(2*(eps**(-2))*c2* (2**((-beta+gamma)*L/2))*  ((1-(2**(-(gamma-beta)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]

                write(logfile, " ".join(["%9d" % n for n in Ns]))
                write(logfile, " \n");
            dNl = np.maximum(0, Ns-Nl)
            write(logfile,'dnl after first pass:')
            write(logfile, " ".join(["%9d" % n for n in dNl]))
            write(logfile, " \n");
            
            # if (almost) converged, estimate remaining error and decide
            # whether a new level is required

            if sum(dNl > 0.01*Nl) == 0:
                write(logfile,'ml')
                write(logfile, " ".join(["%.4f" % n for n in ml]))
                write(logfile, " \n");
                rem = ml[L-Lmin] / (2.0**alpha - 1.0)

                if rem > eps/2:
                    if L == Lmax:
                        print("Warning: Failed to achieve weak convergence")
                    else:
                        write(logfile, 'Nl rem: ')
                        write(logfile, " ".join(["%9d" % n for n in Nl]))
                        write(logfile, "\n")
                        L = L + 1
                        Vl = np.append(Vl, Vl[-1] / 2.0**beta)
                        Nl = np.append(Nl, 0.0)
                        suml = np.column_stack([suml, [0, 0]])

                        Cl = 2**(gamma * np.arange(Lmin, L+1))
                        # recalculate optimum number of samples 
                        if beta > gamma:
                            Ns = [np.ceil(2*(eps**(-2))*c2*((1-(2**(-(beta-gamma)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]
                        else:
                            Ns = [np.ceil(2*(eps**(-2))*c2* (2**((-beta+gamma)*L/2))*  ((1-(2**(-(gamma-beta)/2)))**(-1))*(2**(-(beta+gamma)*i/2))) for i in np.arange(Lmin, L+1)]
                        write(logfile, 'Ns rem: ')
                        write(logfile, " ".join(["%9d" % n for n in Ns]))
                        write(logfile, "\n")
                        dNl = np.maximum(0, Ns-Nl)
                        write(logfile, 'dNl')
                        write(logfile, " ".join(["%9d" % n for n in dNl]))
                        write(logfile, 'end rem')
                        write(logfile, " \n");
        Nslisteps.append(Nl)
        varlevel.append(Vl)
        toc = timeit.default_timer()
        # finally, evaluate the multilevel estimator
        zero_value = next((i for i, x in enumerate(Nl) if x ==0), None)
        P_seq = suml[0,0:zero_value]/Nl[0:zero_value]
        P = sum(suml[0,0:zero_value]/Nl[0:zero_value])
	    
        write(logfile, " ".join(["%9d" % n for n in Nl]))
        write(logfile, " \n")
        write(logfile, "Vl:")
        write(logfile, " ".join(["%.10f" % n for n in Vl]))
        write(logfile, " \n")

        for i in range(0, L+1-Lmin):
            if Nl[i] == min_num:
                print("Warning: Optimum number of samples for level %0d is less than N0 (number used for convergence) so time will not scale" % (i))
                write(logfile, "\n")
                write(logfile, "Warning: Optimum number of samples for level %0d is less than N0 (number used for convergence) so time will not scale" % (i))
                write(logfile, "\n") 
        cost_per_epsilon.append(toc-tic)
        P_list.append(P)
        P_seq_list.append(P_seq)

        write(logfile, "\n") 
        write(logfile, "Expected value: %f " % (P))
        write(logfile, "\nTotal samples at each level l:") 
        write(logfile, " ".join(["%9d" % n for n in Nl]))
        write(logfile, "\n")
            

        
    write(logfile, "\n")
    
    return Nslisteps, varlevel, P_list, P_seq_list, cost_per_epsilon
