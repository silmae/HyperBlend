# HyperBlend version 0.2.0

This is the second released version of HyperBlend on our way towards a full canopy scale vegetation 
simulator. HyperBlend is developed in 
[Spectral Laboratory](https://www.jyu.fi/it/en/research/our-laboratories/hsi) of University of Jyväskylä 
by Kimmo Riihiaho (kimmo.a.riihiaho at jyu.fi). 
You can freely use and modify this software under terms of the MIT licence (see LICENCE file). 
If you use the software, please give us credit for our work by citing our published scientific 
papers (instructions below).

Version 0.2.0 will break many things in the previous version. Don't expect simulations created 
in 0.1.0 to work. Also the folder structure and some constants have been reorganized and renamed. 

Main improvements in this version:

  1. Simulation speed 200 times faster (simulation accuracy decreases 2-4 times) 
     - You can still use the old simulation method if you need maximum accuracy 
  1. Incorporation of the PROSPECT leaf model 
      - You can now use PROSPECT parameters such as water thickness and chlorophyll content
      - It is fairly simple to plug in any other leaf model you would like. Just follow how 
        our local ```prospect``` module does it, and you should be fine
        
Because of bad version controlling on author's part, the master branch has code for the 
canopy model already. I promise you it is hideous spaghetti and should not be used until 
released (probably tagged with v.0.3.0). The canopy code is mostly contained in 
```forest``` and ```blender_scripts``` modules so it should not interfere with normal 
usage of HyperBlend.

### Table of contents 

  1. [About HyperBlend](#About HyperBlend) 
  1. [How to cite](#How to cite)
  1. [Installing](#Installing)
  1. [Working principle](#Working principle)
  1. [Usage](#Usage)
  1. [Project structure](#Project structure)

## <a name="About HyperBlend"></a> About HyperBlend

In recent decades, remote sensing of vegetation by hyperspectral imaging has been of great interest. 
A plant’s growth and development can be negatively affected by biotic or abiotic stress factors. 
Biotic stress is caused by other organisms, such as insects, pathogens, or other plants. Abiotic 
stress refers to physical and chemical factors in growth environment, such as pollution, drought, 
and nutrient availability. High spectral resolution of hyperspectral images enables finding spectral 
bands that are most distinctly affected by a certain stressor.

Analysis of remotely sensed spectral images relies largely on machine learning (ML) algorithms, 
although statistical tests have also been widely applied. As training of ML algorithms requires 
large amounts of training data cumbersome to gather from real-world measurements, it is often 
generated with simulators. In practice, the simulators, or leaf optical properties models, mimic 
spectral response of plants with known biophysical and biochemical properties. Predicting the 
presence and type of stressors from a real hyperspectral image is done by finding spectral features 
that match the simulation, i.e., the simulation problem is inverted.

A rough taxonomy of the leaf optical properties models can be established by examining the scale 
that they operate on: cell scale models depict the intricate internal structure of a plant leaf, 
leaf scale models see the leaves as a whole, and canopy scale models simulate the reflectance of 
groups of. Canopy scale simulators are often statistical extensions of leaf models with some additional 
elements, such as soil reflectance. Canopy scale models can simulate remotely sensed hyperspectral 
images and thus, they are essential in training ML algorithms.

The existing canopy scale models are mostly designed to simulate imagers onboard of satellites or 
airplanes. They have big ground pixel size which allows using approximate geometry or statistical 
values. HyperBlend is one of the rare simulators that will be able to produce intricate vegetation 
geometry as shown in the image below. Detailed geometry is needed when drone-based imaging is 
simulated because the ground pixel size is on the millimeter or centimeter scale and the geometry 
of the plants really matters.

![Preview of HyperBlend's canopy model forest.](readme_img/forest_walker.png){style="width:60%; "}

HyperBlend is currently (v0.2.0) operating at leaf scale level, but preliminary tests on expanding the 
simulator into a canopy scale model have shown promising results. HyperBlend leaf model depicts a leaf 
as a flat box whose volume is filled with absorbing and scattering particles. Desired reflectance and 
transmittance properties can be achieved by adjusting absorbing particle density, scattering particle 
density, forward versus backwards scattering tendency, and mixing factor of the two particle types. 
HyperBlend utilizes 3D-modeling and rendering software Blender and especially its path tracing rendering 
engine Cycles.

In version 0.2.0, we introduce improvements and new features to HyperBlend leaf model. We present two 
methods for increasing simulation speed of the model up to two hundred times faster with slight decrease 
in simulation accuracy. We integrate the well-known PROSPECT leaf model into HyperBlend allowing us to 
use the PROSPECT parametrization for leaf simulation. For the first time, we show that HyperBlend generalizes 
well and can be used to accurately simulate a wide variety of plant leaf spectra.

For more information on the working principle of HyperBlend, see the scientific papers listed below.


##  <a name="How to cite"></a> How to cite

If you find our work usefull in your project, please cite us:

Riihiaho, K. A., Rossi, T., and Pölönen, I.: HYPERBLEND: SIMULATING SPECTRAL REFLECTANCE AND TRANSMITTANCE OF LEAF TISSUE WITH BLENDER, ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., V-3-2022, 471–476, https://doi.org/10.5194/isprs-annals-V-3-2022-471-2022, 2022.

```bibtex
@Article{riihiaho22,
AUTHOR = {Riihiaho, K. A. and Rossi, T. and P\"ol\"onen, I.},
TITLE = {HYPERBLEND: SIMULATING SPECTRAL REFLECTANCE AND TRANSMITTANCE OF LEAF TISSUE WITH BLENDER},
JOURNAL = {ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {V-3-2022},
YEAR = {2022},
PAGES = {471--476},
URL = {https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2022/471/2022/},
DOI = {10.5194/isprs-annals-V-3-2022-471-2022}
}
```

**Citation info for the v0.2.0 paper will be shown here once we get it from the publisher!**

## <a name="Installing"></a> Installing

Clone the repository to some location on your machine. Create a python environment by running 
`conda env create -n hb --file hb_env.yml` in yor anaconda command prompt when in project root directory.
Use your favourite IDE for editing and running the code (developed using PyCharm). 
Command line build and run is untested, but it should work as well.

You will also need open-source 3D-modeling and rendering software Blender, which 
you can download and install from ([blender.org](blender.org)). At least versions 2.8x and 2.9x should 
work (developed on version 2.93.5). Change your Blender executable path to `constants.py`.

## <a name="Working principle"></a> Working principle

The measured (or simulated) reflectances and transmittances look like this:  

| wavelength [nm] | reflectance | transmittance
|---|---|---|
|400 | 0.21435 | 0.26547|
|401 | 0.21431 | 0.26540|
|... | ... | ... |

We call this the *target*. Reflectance and transmittance values represent the fraction of 
reflected and transmitted light so both values are separately bound to closed interval [0,1] 
and their sum cannot exceed 1. 

We use a Blender scene with a rectangular box that represents a leaf (```scene_leaf_material.blend```). 
The material of the leaf has four adjustable parameters: absorption particle density, scattering 
particle density, scattering anisotropy, and mix factor. These control how the light is scattered and 
absorbed in the leaf material. HyperBlend will find the values that will produce the same reflectance 
transmittance spectrum that the target has.

For each wavelength in the target, we adjust the leaf material parameters until the modeled 
reflectance and transmittance match the target values. As of version 0.2.0, there are three 
different ways to do this:

  1. Original optimization method (slow but accurate)
  1. Surface fitting method approximates parameter spaces with analytical function surfaces
  1. NN (neural network) does the same as Surface fitting but uses NN instead of analytical functions 

NN is currently the default and recommended method as it is fast (200 x speedup compared to Original) 
and fairly accurate (error increased 2-4 times compared to Original). Surface fitting is as fast as NN 
but not as accurate.


## <a name="Usage"></a> Usage

## Generating new starting 

The starting guess affects how fast the optimization method can find 
target reflectance and transmittance. Reasonable starting guess is 
included in the repository, but you can fiddle around with it if you want 
to. You might want to rename the old starting guess first, so that it will 
not get overwritten. You can always get the starting guess from the Git repository 
as well. 

Generating new starting for original optimization method can be done by 

```python
from src.utils import spectra_utils as SU

SU.generate_starting_guess()
SU.fit_starting_guess_coefficients()
```

Copy paste this to ```__main__.py``` and run.
New starting guess is automatically used from this point onwards. 
Starting guess is stored in the root folder ```src```.

The entry point of the software is `__main__.py` file. For testing the software without actual data, 
run 

```python
from src import presets

presets.optimize_default_target()
```

that uses hard-coded test spectrum of a leaf. 

The base element of the software is a measurement set identified by `set_name`, which consists 
of one or more samples identified by `sample_id`. To initialize a new set, initialize an 
`Optimization` object which will create a directory structure for given `set_name` under 
`optimization` directory. 

To use real measured data, you should use 

```
data.toml_handling.write_target(set_name:str, data, sample_id=0)
```

where `data` is a list of wavelength, reflectance, transmittance 3-tuples (or lists). This will 
write the data to disk in human-readable toml-formatted form that the rest of the code can understand.

Now you can start the optimization process. To summarize a simple use case in one snippet:

```python
from src.leaf_model.material_param_optimization import Optimization
from data import toml_handlling as TH

data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
set_name = 'test_set'

o = Optimization(set_name)
TH.write_target(set_name, data, sample_id=0)
o.run_optimization()
```

The results are written onto disk in the set's directory as toml files and plotted to .png images.

## <a name="Project structure"></a>  Project structure, *i.e.*, where to find stuff

Descriptions of the most important files.

- `optimization` Optimization results and targets are stored here in set-wise sub-directories.
- `src` Top level source code package.
  - `__main__.py` Entrypoint of the software.
  - `constants.py` Mainly names of things that should not be changed unless you are sure what you are doing. 
  With the exception of path to Blender executable that you have to change to match your installation.
  - `optimization.py` Optimization work is done here. 
  - `plotter.py` Responsible for plotting the results.
  - `presets.py` Default runnable example with hard-coded spectral values.
  - `data` Package responsible for data structure. Making changes in here will likely result in 
  failure to read old optimization results.
    - `file_handling.py` Creation and removal of files and directories. Data structure reduction and expansion 
    for more convenient file sharing.
    - `file_names.py` Knows all filenames in the project. Generator-parser pairs.
    - `path_handling.py` Knows the most important paths used in the project. Some paths may still need 
    to be generated manually.
    - `toml_handling.py` Writing and reading of result data files.
  - `rendering` Package responsible for calling Blender.
  - `utils` Package containing miscellaneous utility modules.
- `bs_render_single.py` Blender render script file.
- `scene_leaf_material.blend` Bender scene file that is run by the `bs_render_single.py`.
