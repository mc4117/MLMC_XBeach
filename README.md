MLMC framework for XBeach
================

This repository contains the model described in the paper

*Mariana C. A. Clare, Matthew D. Piggott and Colin J. Cotter*, **Multilevel Monte Carlo methods for erosion and flood risk assessment in the coastal zone**, Computer Methods in Applied Mechanics and Engineering.


Software requirements
-------------------------

1. XBeach (https://oss.deltares.nl/web/xbeach/)
    * In this work, we used version 1.23.5526 XBeachX release
2. xbeachtools (https://github.com/openearth/xbeach-tools-python/tree/python3)
    * Used to run XBeach through Python
3. Python 3.5 or later


Simulation scripts
------------------

* Section 3.2.1: 1D Test Case
    
    Reproduce MLMC results with:

```
#!bash
    $ python slope_1d_files/slope_1d_mlmc.py
```
    Note use init_test = True for preliminary run and eps_test = True for full run.

    Reproduce MC results with:

```
#!bash
    $ python slope_1d_files/slope_1d_mc.py
```
    Note this file must be run multiple times to achieve a sufficient number of samples.

* Section 3.2.2: 2D Test Case
    
    Reproduce MLMC results with:

```
#!bash
    $ python slope_2d_files/slope_2d_mlmc.py
```
    Note use init_test = True for preliminary run and eps_test = True for full run.

    Reproduce MC results with:

```
#!bash
    $ python slope_2d_files/slope_2d_mc.py
```
    Note this file must be run multiple times to achieve a sufficient number of samples.
   
 * Section 3.2.3: Morphology Test Case
    
    Reproduce MLMC results with:

```
#!bash
    $ python morphological_files/morph_mlmc.py
```
    Note use init_test = True for preliminary run and eps_test = True for full run.

    Reproduce MC results with:

```
#!bash
    $ python  morphological_files/morph_mc.py
```
    Note this file must be run multiple times to achieve a sufficient number of samples.
    
 * Section 3.2.4: Boscombe Beach Test Case
    
    Reproduce MLMC results with:

```
#!bash
    $ python boscombe_files/boscombe_mlmc.py
```
    Note use init_test = True for preliminary run and eps_test = True for full run.

    Reproduce MC results with:

```
#!bash
    $ python  boscombe_files/boscombe_mc.py
```
    Note this file must be run multiple times to achieve a sufficient number of samples.
    
 * Section 4: Cumulative Distribution Function
 
   Reproduce the Cumulative Distribution Functions using the python files in the CDF folder of this repository. Note before these files can be run, it is necessary to generate outputs from the MLMC algorithm (using only one epsilon value) and the MC algorithm, using the files discussed for Section 3.2.    