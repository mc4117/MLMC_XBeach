# Function which sets up XBeach input files, produces the command to run XBeach and then extracts relevant outputs
import numpy as np

import subprocess
from netCDF4 import Dataset
import logging
import os

from bathy import Bathymetry
from xbeachtools import XBeachModel
from xbeachtools import XBeachBathymetry

from scipy import interpolate

def build_xbeach(diroutmain, nf, sigma, variable_name, interp_fn_orig, l_max, l_min = 1):
    
    total_length_x = 1250
    
    # set up parameters for XBeach input
    p = dict(
             #processes
             order = 1,
             bedfriccoef = 55,
             flow=1,
             sedtrans = 0,
             morphology = 0,
             avalanching = 0,
             cyclic = 0,
             wavemodel = 'nonh',
             #grid
             dx = total_length_x/nf,
             dy = 0,
             vardx = 1,
             posdwn = 0,
             xfile = 'x.txt',
             xori = 0,
             yori = 0,
             Hrms = 2.5,
             Trep = 10,
             dir0 = 270,
             instat     = 'stat',
             front = 'nonh_1d',
             #output
             outputformat = 'netcdf',
             tintg = 200,
             tintm = 200,
             tstop = 200,
             nglobalvar = ['zb','zs','hh'],
             nmeanvar = ['zs', 'u'])
             #nx = 2**6)
             
    b = dict(height = 2,
             length_x = total_length_x)
             
    logger = logging.getLogger(__name__)
    logger.info('setup.py is called for')

    # create directory where XBeach will run  
    path = os.path.join(diroutmain, 'test')
    if not os.path.exists(path):
        os.makedirs(path)
    # if less than minimum level considered in algorithm, return 0
    if nf <= l_min:
        maximum_surge = 0
        return maximum_surge
                         
    if variable_name == 'bed_slope':
        xb = XBeachModel(**p)
        bp = b.copy()
        bp.update(p)
        bathy = Bathymetry(**bp)
        bed_slope = sigma
        bathy.mc_slope(b['length_x']/(l_max), 0.6, bed_slope)
        interp_fn = interpolate.interp1d(bathy.x, bathy.z)
        dx = b['length_x']/nf
        x = np.arange(0, b['length_x']+dx, dx)
        z = interp_fn(x)
    else:
        bed_slope = 0.15
        exec("p['" + variable_name + "'] = " + str(sigma))
        dx = b['length_x']/nf
        x = np.arange(0, b['length_x']+dx, dx)
        z = interp_fn_orig(x)
        xb = XBeachModel(**p)
        bp = b.copy()
        bp.update(p)
        bathy = Bathymetry(**bp)
                                                 
    XBbathy = XBeachBathymetry(x, z)
                                             
    logger.debug(XBbathy)
                                                     
    xb['bathymetry'] = XBbathy
                                                 
    os.chdir(path)
                                                         
    xb.write(path)
                  
    # run XBeach
    subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)

    # collect output
    dataset = Dataset('xboutput.nc')
                                                         
    x_array = dataset.variables['globalx'][0]
                                                                 
    a = (dataset.variables['zs_max'][0] - dataset.variables['zb'][0]).data[0][0:len(x_array)-1]
                                                             
    # find point at which the water level stops being above the bed level
    count = len(a[a > 0.00000001])

    maximum_surge = x_array[count-1]/total_length_x
                                                                     
    return maximum_surge
