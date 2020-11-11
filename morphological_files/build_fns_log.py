# Function which sets up XBeach input files, produces the command to run XBeach and then extracts relevant outputs
import numpy as np

import subprocess
from netCDF4 import Dataset
import logging
import os

from xbeachtools import XBeachModel
from xbeachtools import XBeachBathymetry

from scipy import interpolate

def build_morphological_changes(diroutmain, nf, sigma, variable_name, interp_fn, angles_fn, lmax, l_min = 1, instat_type = 'reuse', normalisation_factor = 1):
    
    # set up parameters for XBeach input
    p = dict(
             #processes
             bedfriccoef = 55,
             flow=1,
             carspan    = 0,
             random     = 0,
             #grid
             vardx = 1,
             posdwn = -1,
             xfile = 'x.txt',
             yfile = 'y.txt',
             depfile = 'z.txt',
             xori = 0,
             yori = 0,
             nx = nf-1,
             ny = 2,
             thetamin   = -90,
             thetamax   = 90,
             dtheta     = 180,
             # constants
             g          = 9.810000,
             rho        = 1000,
             C          = 65,
             nuh        = 0.100000,
             nuhfac     = 1,
             CFL        = 0.900000,
             eps        = 0.001000,
             umin       = 0,
             hmin       = 0.001000,
             # waves
             instat = instat_type,
             bcfile = 'waves.txt',
             #'break'      = 3,
             gamma      = 0.500000,
             alpha      = 1,
             n          = 10,
             delta      = 0,
             roller     = 1,
             beta       = 0.100000,
             rfb        = 1,
             zs0 = -4,
             # morphology
             avalanching = 1,
             
             rhos       = 2650,
             por        = 0.400000,
             D50        = 0.000250,
             D90        = 0.000375,
             waveform   = 'vanthiel',
             form       = 2,
             facua      = 0.100000,
             turb       = 2,
             Tsmin      = 1,
             morfac     = 10,
             morstart   = 0,
             wetslp     = 0.100000,
             dryslp     = 1,
             hswitch    = 0.100000,
             dzmax      = 0.044100,
             #output
             outputformat = 'netcdf',
             tintg = 100000,
             tintm = 100000,
             tstop = 100000,
             nglobalvar = ['zb','zs','H'],
             nmeanvar = ['zs'])

             
    logger = logging.getLogger(__name__)
    logger.info('setup.py is called for')
             
    # create directory where XBeach will run          
    path = os.path.join(diroutmain, 'test')#_', str(i))
             
    if not os.path.exists(path):
        os.makedirs(path)

    # if less than minimum level considered in algorithm, return 0
    if nf <= l_min:
        morphological_change = 0
        return morphological_change
                         
    xb = XBeachModel(**p)

                         
    os.chdir(path)
                             
                             
    if instat_type != 'reuse':
                                 
        file_list = os.listdir(os.getcwd())

        for item in file_list:
            # remove previous boundary conditions
            if item.endswith(".bcf"):
                os.remove(item)
                                             
        if variable_name == 'Hm0':
            # generate waves file
            open('waves.txt', 'w')
            durationlist = ['1.8316e+004 ', '2.2225e+004 ', '1.9619e+004 ', '2.4861e+004 ', '2.6150e+004 ']
            tplist = [8.8106, 8.8106, 9.1408, 10.1729, 10.1729]
            for i in range(len(durationlist)):
                w = [sigma, tplist[i], 270, 1.0, 20, durationlist[i], 1.0]
                                                     
                with open('waves.txt', 'a+') as fp:
                    fp.write('%0.6f ' %w[0])
                    fp.write('%0.6f ' %w[1])
                    fp.write('%0.6f ' %w[2])
                    fp.write('%0.6f ' %w[3])
                    fp.write('%0.6f ' %w[4])
                    fp.write(w[5])
                    fp.write('%0.6f ' %w[6])
                    fp.write('\n')
                    fp.close()
        else:
            print(variable_name)
            raise ValueError("Need to specify variable name")
                                                                     
                                                                     
    y1 = np.linspace(0, 10, 3)
                                                                 
    xnew = np.linspace(interp_fn.x_min, interp_fn.x_max, int(nf))                                                                                                          
    znew = interp_fn(xnew, y1)
    
    # create bedfile for new x and y grid spacing
    init_bed = znew[0]
                                                                             
    X,Y = np.meshgrid(xnew, y1)
    XBbathy = XBeachBathymetry(X, Y, znew)
                                                                         
    logger.debug(XBbathy)
                                                                                 
    xb['bathymetry'] = XBbathy
                                                                             
                                                                             
    xb.write(path)
                                                                                     
    fp = open("params.txt", "a+")
    fp.write("break = 3")
    fp.close()
    
    # run xbeach                                                                           
    try:
        subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)
    except subprocess.CalledProcessError as e:
        print('Error')
        print(sigma)
        return np.nan

    dataset = Dataset('xboutput.nc')
    
    # use trapezium rule to calculate the difference between final and initial bed and hence amount of eroded material
    final_bed = dataset.variables['zb'][len(dataset.variables['zb'])-1].data[0]
    diff_list = []
    for i in range(len(final_bed)):
        if final_bed[i] - float(init_bed[i]) < 0:
            diff_list.append(final_bed[i] - float(init_bed[i]))
        else:
            diff_list.append(0)

    morphological_change = diff_list[0] + diff_list[-1]

    for i in range(1, len(diff_list)-1):
        morphological_change += 2*diff_list[i]


    morphological_change_final = 0.5*(1/nf)*morphological_change*(interp_fn.x_max - interp_fn.x_min)

    return morphological_change_final/normalisation_factor


