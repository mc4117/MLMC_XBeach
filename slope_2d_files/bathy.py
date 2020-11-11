# Bathymetry class used to generate bathymetry

import numpy as np
from xbeachtools import XBeachBathymetry


class Bathymetry(XBeachBathymetry):  
    def __init__(self, dx= 1, dy= 1, length_x= 0, length_y= 0, **kwargs):  
    
        self.x = None 
        self.y = None 
        self.z = None 
        
        self.dx = dx
        self.dy = dy
        self.length_x = length_x
        self.length_y = length_y
    
    # useful functions
    def yuniform(self, x, z):
        print(self.length_y)
        y = np.arange(0, self.length_y+self.dy, self.dy)
        X,Y = np.meshgrid(x, y)
        print(np.shape(z))
        Z = np.tile(z, (len(y),1))              
        return X, Y, Z
    
    def xuniform(self, y, z):
        x = np.arange(0, self.length_x+self.dx, self.dx)
        X,Y = np.meshgrid(x, y)
        Z = np.tile(z, (len(x),1)).transpose()              
        return X, Y, Z
    
    # specific bathymetry profiles
        
    def mc_slope(self, dx, ratio_slope, bed_slope):
        
        x = np.arange(0, self.length_x+dx, dx)
        
        bath = []
	
        for j in range(len(x)):
            if j < ratio_slope*len(x) -1:

                bath.append(-15)
            else:
                bath.append(bed_slope*(x[j] - x[int(np.floor(ratio_slope*len(x)-1))]) -15)

        self.x = x
        self.z = np.array(bath)
        

    def mc_slope_test(self, dx, dy, ratio_slope, bed_slope_x, bed_slope_y):
        x = np.arange(0, self.length_x+dx, dx)
        y = np.arange(0, self.length_y+dy, dy)
        bath = []
        for i in range(len(y)):
            for j in range(len(x)):
                if j < ratio_slope*len(x)-1 and i < ratio_slope*len(y)-1:
                    bath.append(-15)
                elif j >= ratio_slope*len(x)-1 and i < ratio_slope*len(y)-1:
                    bath.append(bed_slope_x*(x[j] - x[int(np.floor(ratio_slope*len(x)-1))]) -15)
                elif j < ratio_slope*len(x) -1 and i >= ratio_slope*len(y)-1:
                    bath.append(bed_slope_y*(y[i] - y[int(np.floor(ratio_slope*len(y)-1))]) -15)
                else:
                    bath.append(bed_slope_x*(x[j] - x[int(np.floor(ratio_slope*len(x)-1))]) -15 + bed_slope_y*(y[i] - y[int(np.floor(ratio_slope*len(y)-1))]))
    
        z = np.array(bath)

        self.x = x
        self.y = y
        self.z = z
