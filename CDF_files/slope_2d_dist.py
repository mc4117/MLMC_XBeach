#####
# Calculate MLMC and MC cumulative distribution function for slope 2d test case
#
# Note no output files have been included in this repository so all csv files must be produced using 
# the files in the slope_2d_files folder
#####

import pandas as pd
import pylab as plt
import numpy as np
from scipy import ndimage


## Read in monte carlo data
output = pd.read_csv('slope_2d_mc/total_output.csv')


## Read in first sample data and sort it 
output2 = pd.read_csv('slope_2d_interp_ag/output2.csv').sort_values('output').reset_index(drop = True)['output']
output3 = pd.read_csv('slope_2d_interp_ag/output3.csv').sort_values('output').reset_index(drop = True)['output']
output4 = pd.read_csv('slope_2d_interp_ag/output4.csv').sort_values('output').reset_index(drop = True)['output']
output5 = pd.read_csv('slope_2d_interp_ag/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6 = pd.read_csv('slope_2d_interp_ag/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7 = pd.read_csv('slope_2d_interp_ag/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8 = pd.read_csv('slope_2d_interp_ag/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9 = pd.read_csv('slope_2d_interp_ag/output9.csv').sort_values('output').reset_index(drop = True)['output']


## Read in second sample data and sort it
output2_2 = pd.read_csv('slope_2d_interp_ag2/output2.csv').sort_values('output').reset_index(drop = True)['output']
output3_2 = pd.read_csv('slope_2d_interp_ag2/output3.csv').sort_values('output').reset_index(drop = True)['output']
output4_2 = pd.read_csv('slope_2d_interp_ag2/output4.csv').sort_values('output').reset_index(drop = True)['output']
output5_2 = pd.read_csv('slope_2d_interp_ag2/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6_2 = pd.read_csv('slope_2d_interp_ag2/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7_2 = pd.read_csv('slope_2d_interp_ag2/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8_2 = pd.read_csv('slope_2d_interp_ag2/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9_2 = pd.read_csv('slope_2d_interp_ag2/output9.csv').sort_values('output').reset_index(drop = True)['output']


# sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]

# create inverse samples
x_ilist = []

for i in range(1000000):
    x_i = 0
    u = unif()
    
    x_i += output3_2[np.floor(u*len(output3_2))]

    x_i += output4_2[np.floor(u*len(output4_2))] - output3[np.floor(u*len(output3))]

    x_i += output5_2[np.floor(u*len(output5_2))] - output4[np.floor(u*len(output4))]

    x_i += output6_2[np.floor(u*len(output6_2))] - output5[np.floor(u*len(output5))]

    x_i += output7_2[np.floor(u*len(output7_2))] - output6[np.floor(u*len(output6))]

    x_i += output8_2[np.floor(u*len(output8_2))] - output7[np.floor(u*len(output7))]

    x_i += output9_2[np.floor(u*len(output9_2))] - output8[np.floor(u*len(output8))]

    x_ilist.append(x_i)

# remove unphysical results    
x_phys = []

for i in x_ilist:
    if i > 0.62890625:
        if i <= 1:
            x_phys.append(i)

# plot pdfs        
output_converted = [i for i in output['output']]     

cleaned_x_ilist = [x for x in x_ilist if str(x) != 'nan']
cleanedList = [x for x in output['output'] if str(x) != 'nan']       
            
plt.hist(cleanedList, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(cleaned_x_ilist, bins = 101, histtype=u'step', density = True, label = 'MLMC')
plt.hist(x_phys, bins = 101, histtype = u'step', density = True, label = 'physical MLMC')
plt.legend()
plt.show()            

hist_out, bins_out = np.histogram(np.asarray(cleanedList), bins = 101, density = True)
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