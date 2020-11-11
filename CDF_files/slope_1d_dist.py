#####
# Calculate MLMC and MC cumulative distribution function for slope 1d test case
#
# Note no output files have been included in this repository so all csv files must be produced using 
# the files in the slope_1d_files folder
#####

import pandas as pd
import pylab as plt
import numpy as np
from scipy import ndimage


## Read in monte carlo data
output = pd.read_csv('slope_1d_mc/total_output.csv')
output_converted = [i/1250 for i in output['output']]

## Read in first sample data and sort it 
output2 = pd.read_csv('slope_1d_interp_ag_sing/output2.csv').sort_values('output').reset_index(drop = True)['output']
output3 = pd.read_csv('slope_1d_interp_ag_sing/output3.csv').sort_values('output').reset_index(drop = True)['output']
output4 = pd.read_csv('slope_1d_interp_ag_sing/output4.csv').sort_values('output').reset_index(drop = True)['output']
output5 = pd.read_csv('slope_1d_interp_ag_sing/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6 = pd.read_csv('slope_1d_interp_ag_sing/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7 = pd.read_csv('slope_1d_interp_ag_sing/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8 = pd.read_csv('slope_1d_interp_ag_sing/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9 = pd.read_csv('slope_1d_interp_ag_sing/output9.csv').sort_values('output').reset_index(drop = True)['output']
output10 = pd.read_csv('slope_1d_interp_ag_sing/output10.csv').sort_values('output').reset_index(drop = True)['output']
output11 = pd.read_csv('slope_1d_interp_ag_sing/output11.csv').sort_values('output').reset_index(drop = True)['output']
output12 = pd.read_csv('slope_1d_interp_ag_sing/output12.csv').sort_values('output').reset_index(drop = True)['output']
output13 = pd.read_csv('slope_1d_interp_ag_sing/output13.csv').sort_values('output').reset_index(drop = True)['output']

## Read in second sample data and sort it
output2_2 = pd.read_csv('slope_1d_interp_ag_sing2/output2.csv').sort_values('output').reset_index(drop = True)['output']
output3_2 = pd.read_csv('slope_1d_interp_ag_sing2/output3.csv').sort_values('output').reset_index(drop = True)['output']
output4_2 = pd.read_csv('slope_1d_interp_ag_sing2/output4.csv').sort_values('output').reset_index(drop = True)['output']
output5_2 = pd.read_csv('slope_1d_interp_ag_sing2/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6_2 = pd.read_csv('slope_1d_interp_ag_sing2/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7_2 = pd.read_csv('slope_1d_interp_ag_sing2/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8_2 = pd.read_csv('slope_1d_interp_ag_sing2/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9_2 = pd.read_csv('slope_1d_interp_ag_sing2/output9.csv').sort_values('output').reset_index(drop = True)['output']
output10_2 = pd.read_csv('slope_1d_interp_ag_sing2/output10.csv').sort_values('output').reset_index(drop = True)['output']
output11_2 = pd.read_csv('slope_1d_interp_ag_sing2/output11.csv').sort_values('output').reset_index(drop = True)['output']
output12_2 = pd.read_csv('slope_1d_interp_ag_sing2/output12.csv').sort_values('output').reset_index(drop = True)['output']
output13_2 = pd.read_csv('slope_1d_interp_ag_sing2/output13.csv').sort_values('output').reset_index(drop = True)['output']


# sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]

# create inverse samples
x_ilist = []

for i in range(1000000):
    x_i = 0
    u = unif()
    
    x_i += output4_2[np.floor(unif()*len(output4_2))]

    x_i += output5_2[np.floor(u*len(output5_2))] - output4[np.floor(u*len(output4))]

    x_i += output6_2[np.floor(u*len(output6_2))] - output5[np.floor(u*len(output5))]

    x_i += output7_2[np.floor(u*len(output7_2))] - output6[np.floor(u*len(output6))]

    x_i += output8_2[np.floor(u*len(output8_2))] - output7[np.floor(u*len(output7))]

    x_i += output9_2[np.floor(u*len(output9_2))] - output8[np.floor(u*len(output8))]    

    x_i += output10_2[np.floor(u*len(output10_2))] - output9[np.floor(u*len(output9))]    

    x_i += output11_2[np.floor(u*len(output11_2))] - output10[np.floor(u*len(output10))]    

    x_i += output12_2[np.floor(u*len(output12_2))] - output11[np.floor(u*len(output11))] 

    x_i += output13_2[np.floor(u*len(output13_2))] - output12[np.floor(u*len(output12))] 

    x_ilist.append(x_i)

# remove unphysical results    
x_phys = []

for i in x_ilist:
    if i > 0.6282958984375999:
        if i <= 1:
            x_phys.append(i)


# plot pdfs            
plt.hist(output_converted, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(x_ilist, bins = 101, histtype=u'step', density = True, label = 'MLMC')
plt.hist(x_phys, bins = 101, histtype = u'step', density = True, label = 'physical MLMC')
plt.legend()
plt.show()            

hist_out, bins_out = np.histogram(np.asarray(output_converted), bins = 101, density = True)
hist, bins = np.histogram(np.asarray(x_phys), bins = 101, density = True)

# calculate cumulative distributions
cum_hist_out = np.cumsum(hist_out)
cum_hist = np.cumsum(hist)

plt.plot(bins[1:], cum_hist/cum_hist[-1], label = 'MLMC')
plt.plot(bins_out[1:], cum_hist_out/cum_hist_out[-1], label = 'MC') 
plt.legend()
plt.show()
           
a = cum_hist/cum_hist[-1]

b = cum_hist_out/cum_hist_out[-1]

error = sum([(a[i] - b[i])**2 for i in range(len(a))])/(sum([b[i]**2 for i in range(len(b))]))

print(error)