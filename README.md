# ILSI
Package for iterative linear stress inversion (ILSI) of focal mechanism data and slickenside data. Documentation at [https://ebeauce.github.io/ILSI/](https://ebeauce.github.io/ILSI/)<br>

## Reference
Please refer the following article if you use ILSI for your research:<br>
Eric Beaucé, Robert D. van der Hilst, Michel Campillo. An Iterative Linear Method with Variable Shear Stress Magnitudes for Estimating the Stress Tensor from Earthquake Focal Mechanism Data: Method and Examples, *Bulletin of the Seismological Society of America*, 2022. DOI: [https://doi.org/10.1785/0120210319](https://doi.org/10.1785/0120210319)

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

    # get solution from original data set (output is a dictionary)
    inversion_output = ILSI.ilsi.inversion_one_set_instability(
            strikes, dips, rakes
        )
    # check out the content of the dictionary
    print(inversion_output.keys())
    # estimate uncertainties by resampling the data set (output is a dictionary)
    shape_ratio = ILSI.utils_stress.R_(principal_stresses)
    bootstrap_output = ILSI.ilsi.inversion_bootstrap_instability(
            inversion_output["principal_directions"],
            shape_ratio,
            strikes,
            dips,
            rakes,
            inversion_output["friction_coefficient"]
        )
    # check out the content of the dictionary
    print(bootstrap_output.keys())
```

### Inverting slickenside data

```python

    import ILSI

    # read strikes, dips and rakes from your data file.

    # get solution from original data set
    inversion_output = ILSI.ilsi.iterative_linear_si(
            strikes, dips, rakes
        )
    # if you want to bootstrap the data set, simply resample the data
    # set successively in a loop and call ILSI.ilsi.iterative_linear_si
    # on the resampled strikes, dips and rakes
```

## Updates

- v1.1.3: Moore-Penrose inverse is supported for variable shear stress.

- v1.1.2: Parallelization is now made with the `n_threads` key-word argument (parallel is still supported but is deprecated).

- v1.1.1: Outputs are now given as dictionaries for more flexibility. Make sure
  to update your scripts.

## Tutorials
Check out the *tests* folder for example scripts. Jupyter notebooks can be run to reproduce the results shown in Beaucé et al. 2022. See the list of required packages at the top of the scripts. The `plotFMC` and `functionsFMC` librairies are from the FMC repository available at [https://github.com/Jose-Alvarez/FMC](https://github.com/Jose-Alvarez/FMC).

## Questions?

If you encounter issues or difficulties with ILSI, don't hesitate to open a new issue or pull request on Github, or contact me at: ebeauce@ldeo.columbia.edu
