#####
# Calculate MLMC and MC cumulative distribution function for morphology test case
#
# Note no output files have been included in this repository so all csv files must be produced using 
# the files in the morphological_files folder
#####

import pandas as pd
import pylab as plt
import numpy as np
from scipy import ndimage


## Read in monte carlo data
output = pd.read_csv('morphological_mc/total_output.csv')

## Read in first sample data and sort it 
output6 = pd.read_csv('morphological_ero_ag_fine/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7 = pd.read_csv('morphological_ero_ag_fine/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8 = pd.read_csv('morphological_ero_ag_fine/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9 = pd.read_csv('morphological_ero_ag_fine/output9.csv').sort_values('output').reset_index(drop = True)['output']
output10 = pd.read_csv('morphological_ero_ag_fine/output10.csv').sort_values('output').reset_index(drop = True)['output']
output11 = pd.read_csv('morphological_ero_ag_fine/output11.csv').sort_values('output').reset_index(drop = True)['output']
output12 = pd.read_csv('morphological_ero_ag_fine/output12.csv').sort_values('output').reset_index(drop = True)['output']


## Read in second sample data and sort it
output6_2 = pd.read_csv('morphological_ero_ag_fine2/output6.csv').sort_values('output').reset_index(drop = True)['output']
output7_2 = pd.read_csv('morphological_ero_ag_fine2/output7.csv').sort_values('output').reset_index(drop = True)['output']
output8_2 = pd.read_csv('morphological_ero_ag_fine2/output8.csv').sort_values('output').reset_index(drop = True)['output']
output9_2 = pd.read_csv('morphological_ero_ag_fine2/output9.csv').sort_values('output').reset_index(drop = True)['output']
output10_2 = pd.read_csv('morphological_ero_ag_fine2/output10.csv').sort_values('output').reset_index(drop = True)['output']
output11_2 = pd.read_csv('morphological_ero_ag_fine2/output11.csv').sort_values('output').reset_index(drop = True)['output']
output12_2 = pd.read_csv('morphological_ero_ag_fine2/output12.csv').sort_values('output').reset_index(drop = True)['output']

# sample function for u
def unif():
    return np.random.uniform(0, 1.0, 1)[0]

# create inverse sample
x_ilist = []

for i in range(1000000):
    x_i = 0
    u = unif()
    
    x_i += output6_2[np.floor(u*len(output6_2))]

    x_i += output7_2[np.floor(u*len(output7_2))] - output6[np.floor(u*len(output6))]

    x_i += output8_2[np.floor(u*len(output8_2))] - output7[np.floor(u*len(output7))]
    
    x_i += output9_2[np.floor(u*len(output9_2))] - output8[np.floor(u*len(output8))]

    x_i += output10_2[np.floor(u*len(output10_2))] - output9[np.floor(u*len(output9))]

    x_i += output11_2[np.floor(u*len(output11_2))] - output10[np.floor(u*len(output10))]

    x_i += output12_2[np.floor(u*len(output12_2))] - output11[np.floor(u*len(output11))]

    x_ilist.append(x_i)

# add finest level outputs to inverse sample list 
# (these are the only true finest level values that can be included from an MLMC method)
for i in range(len(output12)):
    x_ilist.append(output12[i])
    
for i in range(len(output12_2)):
    x_ilist.append(output12_2[i])  

# plot pdfs        
output_converted = [i for i in output['output']]

cleaned_x_ilist = [x for x in x_ilist if str(x) != 'nan']
cleanedList = [x for x in output['output'] if str(x) != 'nan']     
            
plt.hist(cleanedList, bins = 101, histtype=u'step', density = True, label = 'Monte Carlo')
plt.hist(cleaned_x_ilist, bins = 101, histtype=u'step', density = True, label = 'MLMC')
plt.legend()
plt.show()

hist_out, bins_out = np.histogram(np.asarray(cleanedList), bins = 101, density = True)
hist, bins = np.histogram(np.asarray(cleaned_x_ilist), bins = 101, density = True)

## calculate cumulative distributions
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