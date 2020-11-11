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

def build_xbeach_2d(diroutmain, nf, sigma, variable_name, interp_fn_orig, l_max, l_min = 1):

    total_length_x = 1250
    total_length_y = 1000
    
    # set up parameters for XBeach inputs
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
	    front = 'nonh_1d',
        #grid
        dx = total_length_x/nf,     
        dy = total_length_y/nf,
        vardx = 1,
        posdwn = 0,
        xfile = 'x.txt',
        xori = 0,
        yori = 0,
        Hrms = 2.5,
        Trep = 10,
        dir0 = 270,
        instat     = 'stat',
        #output
        outputformat = 'netcdf',
        tintg = 200,  
        tintm = 200,    
        tstop = 200,
        nglobalvar = ['zb','zs','hh'],
        nmeanvar = ['zs'])

    b = dict(height = 2,
         length_x = total_length_x,
         length_y = total_length_y)

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

    if variable_name == 'bed_slope_2d':
        xb = XBeachModel(**p)
        bp = b.copy()
        bp.update(p)
        bathy = Bathymetry(**bp)
        bed_slope_x = sigma
        bed_slope_y = sigma
        bathy.mc_slope_test(b['length_x']/(l_max), b['length_y']/(l_max), 0.6, bed_slope_x, bed_slope_y)
        interp_fn = interpolate.interp2d(bathy.x, bathy.y, bathy.z)
        dx = b['length_x']/nf
        dy = b['length_y']/nf
        x = np.arange(0, b['length_x']+dx, dx)
        y = np.arange(0, b['length_y']+dy, dy)
        z = interp_fn(x, y)
    else:
        bed_slope_x = 0.15
        bed_slope_y = 0.15
        exec("p['" + variable_name + "'] = " + str(sigma))
        dx = b['length_x']/nf
        dy = b['length_y']/nf
        x = np.arange(0, b['length_x']+dx, dx)
        y = np.arange(0, b['length_y']+dy, dy)
        z = interp_fn_orig(x, y)
        xb = XBeachModel(**p)
        bp = b.copy()
        bp.update(p)
        bathy = Bathymetry(**bp)

    X,Y = np.meshgrid(x, y)
    XBbathy = XBeachBathymetry(X, Y, z)

    logger.debug(XBbathy)

    xb['bathymetry'] = XBbathy

    os.chdir(path)

    xb.write(path)
    
    # run XBeach
    try:
        subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)
    except subprocess.CalledProcessError as e:
        print('Error')
        print(sigma)
        return np.nan
    
    # collect output
    dataset = Dataset('xboutput.nc')
        
    x_array = dataset.variables['globalx'][0]
        
    count = []

    # find point at which the water level stops being above the bed level
    for i in range(len(dataset.variables['zs_max'][0])):
        a = (dataset.variables['zs_max'][0][i] - dataset.variables['zb'][0][i]).data[0:len(x_array)-1]
        count.append(len(a[a > 0.00000001]))

    # maximize over y direction
    count_max = max(count)
    maximum_surge = x_array[count_max-1]/total_length_x
    
    return maximum_surge
