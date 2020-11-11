# Function which sets up XBeach input files, produces the command to run XBeach and then extracts relevant outputs
import numpy as np

import subprocess
from netCDF4 import Dataset
import logging
import os

from xbeachtools import XBeachModel
from xbeachtools import XBeachBathymetry

from scipy import interpolate
from scipy import ndimage


def build_xbeach_boscombe_conv(diroutmain, nf, sigma, variable_name, interp_fn, angles_fn, l_max, l_min = 1, instat_type = 'reuse'):
    
    # set up parameters for XBeach input
    p = dict(
        #processes
        sedtrans = 0,
        morphology = 0,
        #cyclic = 0,     
        wavemodel = 'surfbeat',
        #grid
        vardx = 1,
        nx = nf-1,
        ny = nf-1,
        posdwn = 0,
        xfile = 'x.txt',
        yfile = 'y.txt',
        depfile = 'z.txt',
        xori = 0,
        yori = 0,
        #waves
        zs0 = 0,
        gamma = 0.5,
        alpha = 1,
        n = 10,
        delta = 0,
        roller = 1,
        beta = 0.1,
        rfb = 1,
        thetamax = 90,
        thetamin = -90,
        dtheta = 20,
        wbctype = instat_type,
        random = 0,
        bcfile = 'waves.txt',
        #output
        outputformat = 'netcdf',
        tintg = 4000,
        tintm = 4000,
        tstop = 4000,
        nglobalvar = ['zb','zs','H'],
        nmeanvar = ['zs', 'H'])
        
    b = dict(height = 2)
             
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
                         
    os.chdir(path)

                         
    if instat_type != 'reuse':
        # create waves file with Hm0 being the uncertain height                     
        if variable_name == 'Hm0':
            w = [sigma, 8.7890, 270, 3.3333, 20, p['tstop'], 1]
            ## dictionary for wave input
            with open('waves.txt', 'w') as fp:
                fp.write('%0.6f ' %w[0])
                fp.write('%0.6f ' %w[1])
                fp.write('%0.6f ' %w[2])
                fp.write('%0.6f ' %w[3])
                fp.write('%0.6f ' %w[4])
                fp.write('%0.6f ' %w[5])
                fp.write('%0.6f ' %w[6])
                fp.close()
        else:
            print(variable_name)
            raise ValueError("Need to specify variable name")
            
            
    x = np.linspace(interp_fn.x_min, interp_fn.x_max, int(l_max))
    y = np.linspace(interp_fn.y_min, interp_fn.y_max, int(l_max))

    z = interp_fn(x, y)        
    
    # convolve bathymetry
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
        
    z_new = interp_new_fn(x_new, y_new)    
                                                    
    xb = XBeachModel(**p)
                                                         
    bp = b.copy()
    bp.update(p)
    bathy = Bathymetry(**bp)
    
    # create x.grd, y.grd and bathymetry file which XBeach uses as inputs
    
    X,Y = np.meshgrid(x_new, y_new)
    XBbathy = XBeachBathymetry(X, Y, z_new)
                                                         
    logger.debug(XBbathy)
                                                                 
    xb['bathymetry'] = XBbathy
                                                                 
    xb.write(path)
    
    # run XBeach
    try:
        subprocess.check_call("/rds/general/user/mc4117/home/trunk/src/xbeach/xbeach > /dev/null", shell=True)
    except subprocess.CalledProcessError as e:
        print('Error')
        print(sigma)
        return np.nan
                                                                                     
    # extract XBeach output
    dataset = Dataset('xboutput.nc')
                                                                                 
    x_array = dataset.variables['globalx'][0].data
                                                                                         
    max_runuplist = []
                                                                                     
    w = [8.7890]
    dataarray = dataset.variables['zb'][len(dataset.variables['zb'])-1].data
    angle_form = angles_fn[nf]
    for i in range(len(dataarray)):
        if (dataarray[i] > 0).any() == True:
            # calculate runup height using EuroTop formula
            angle = angle_form(dataset.variables['globaly'][i].data[0])
            hm0 = 0
            for j in range(len(dataset.variables['H_max'])):
                h = interpolate.interp1d(dataarray[i], dataset.variables['H_max'][j].data[i])
                hm0 = max(np.sqrt(2)*h(0), hm0)
            L0 = 9.81*((w[0]/1.1)**2)/(2*np.pi)
            xi_m1 = angle*np.sqrt(L0/hm0)
            max_runuplist.append(hm0*1.65*xi_m1)

    if len(max_runuplist) == 0:
        raise ValueError('No positive values')
    dataset.close()                                                                                     
    
    # record maximum Ru2%
    maximum_runup = max(max_runuplist)
    return maximum_runup
