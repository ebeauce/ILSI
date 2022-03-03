# ILSI
Package for iterative linear stress inversion (ILSI) of focal mechanism data and slickenside data. Documentation at [https://ebeauce.github.io/ILSI/](https://ebeauce.github.io/ILSI/)<br>

## Reference
Please refer the following article if you use ILSI for your research (I will update the reference when the paper will be accepted):<br>
Eric Beaucé, Robert D. van der Hilst, Michel Campillo. An Iterative Linear Method for Estimating the Stress Tensor from Earthquake Focal Mechanism Data: Method and Examples. (Being modified and will be submitted to BSSA). Preprint on ESSOAr: [doi.org/10.1002/essoar.10507639.1](doi.org/10.1002/essoar.10507639.1)

## Installation

### Method 1)

    pip install git+https://github.com/ebeauce/ILSI

### Method 2)

Download the source code from Github and, from root folder (where this README.md file is located), run:
    
    pip install .

The two methods will install ILSI equally well. Method 1 only requires one command line. Method 2 requires you to first download the source code, but at least you can easily retrieve the Jupyter notebooks provided in the *tests* folder.

## Quick example

### Inverting focal mechanisms

```python

    import ILSI

    # read strikes, dips and rakes from your data file.

    # get solution from original data set
    stress_tensor, friction, principal_stresses, principal_directions =\
                  ILSI.ilsi.inversion_one_set_instability(strikes, dips, rakes)
    # estimate uncertainties by resampling the data set
    shape_ratio = ILSI.utils_stress.R_(principal_stresses)
    boot_stress_tensor, boot_principal_stresses, boot_principal_directions =\
                  ILSI.ilsi.inversion_bootstrap_instability(
                        principal_directions, shape_ratio, strikes, dips, rakes, friction)
```

### Inverting slickenside data

```python

    import ILSI

    # read strikes, dips and rakes from your data file.

    # get solution from original data set
    stress_tensor, principal_stresses, principal_directions =\
                 ILSI.ilsi.iterative_linear_si(strikes, dips, rakes)
    # if you want to bootstrap the data set, simply resample the data
    # set successively in a loop and call ILSI.ilsi.iterative_linear_si
    # on the resampled strikes, dips and rakes
```

## Tutorials
Check out the *tests* folder for example scripts. Jupyter notebooks can be run to reproduce the results shown in Beaucé et al. 2021. You might need to install some extra packages to run the examples (*e.g.* seaborn or h5py).
