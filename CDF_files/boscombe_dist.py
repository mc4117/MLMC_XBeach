#####
# Calculate MLMC and MC cumulative distribution function for boscombe beach test case
#
# Note no output files have been included in this repository so all csv files must be produced using 
# the files in the boscombe_files folder
#####

import pandas as pd
import pylab as plt
import numpy as np
from scipy import ndimage


## Read in monte carlo data
output = pd.read_csv('bocombe_cdt_2_mc_full/total_output.csv')

## Read in first sample data and sort it 
output5 = pd.read_csv('bocombe_conv_1_small2/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6 = pd.read_csv('bocombe_conv_1_small2/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7 = pd.read_csv('bocombe_conv_1_small2/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8 = pd.read_csv('bocombe_conv_1_small2/output8.csv').sort_values('output').reset_index(drop = True)['output']

## Read in second sample data and sort it
output5_2 = pd.read_csv('bocombe_conv_1_small3/output5.csv').sort_values('output').reset_index(drop = True)['output']
output6_2 = pd.read_csv('bocombe_conv_1_small3/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7_2 = pd.read_csv('bocombe_conv_1_small3/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8_2 = pd.read_csv('bocombe_conv_1_small3/output8.csv').sort_values('output').reset_index(drop = True)['output']

## Sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]


# create inverse samples
x_ilist = []

for i in range(30000):
    x_i = 0
    u = unif()
    
    x_i += output5_2[np.floor(u*len(output5_2))]
    
    x_i += output6_2[np.floor(u*len(output6_2))] - output5[np.floor(u*len(output5))]

    x_i += output7_2[np.floor(u*len(output7_2))] - output6[np.floor(u*len(output6))]
    
    x_i += output8_2[np.floor(u*len(output8_2))] - output7[np.floor(u*len(output7))]

    x_ilist.append(x_i)

# add finest level outputs to inverse sample list 
# (these are the only true finest level values that can be included from an MLMC method)
for i in range(len(output8)):
    x_ilist.append(output8[i])
    
for i in range(len(output8_2)):
    x_ilist.append(output8_2[i])  
    
# plot pdfs        
output_converted = [i for i in output['output']]

cleaned_x_ilist = [x for x in x_ilist if str(x) != 'nan']
cleanedList = [x for x in output_converted if str(x) != 'nan']     
            
plt.hist(cleanedList, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(cleaned_x_ilist, bins = 101, histtype=u'step', density = True, label = 'MLMC')
plt.legend()
plt.show()

hist_out, bins_out = np.histogram(np.asarray(cleanedList), bins = 101, density = True)
hist, bins = np.histogram(np.asarray(cleaned_x_ilist), bins = 101, density = True)

## calculate cumulative distributions
cum_hist_out = np.cumsum(hist_out)
cum_hist = np.cumsum(hist)

plt.plot(bins[1:], cum_hist/cum_hist[-1], label = 'orig MLMC')
plt.plot(bins_out[1:], cum_hist_out/cum_hist_out[-1], label = 'orig MC') 

plt.legend()
plt.show()

plt.plot(bins[1:], cum_hist/cum_hist[-1], label = 'MLMC')
plt.plot(bins_out[1:], cum_hist_out/cum_hist_out[-1], label = 'MC') 
plt.legend()
plt.show()
           
a = cum_hist/cum_hist[-1]

b = cum_hist_out/cum_hist_out[-1]

error = sum([(a[i] - b[i])**2 for i in range(len(a))])/(sum([a[i]**2 for i in range(len(a))]))

print(error)

# as a test remove all very small values from MC outputs which are not covered in the MLMC range
out_list = []
for i in range(len(output_converted)):
    if output_converted[i] >= min(x_ilist):
        out_list.append(output_converted[i])

hist_out_red, bins_out_red = np.histogram(np.asarray(out_list), bins = 101, density = True)

cum_hist_out_red = np.cumsum(hist_out_red)

b = cum_hist_out_red/cum_hist_out_red[-1]

error_2 = sum([(a[i] - b[i])**2 for i in range(len(a))])/(sum([b[i]**2 for i in range(len(b))]))

print(error_2)