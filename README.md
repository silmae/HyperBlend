# HyperBlend version 0.1.0

This is the HyperBlend leaf spectra simulator developed in 
[Spectral Laboratory](https://www.jyu.fi/it/en/research/our-laboratories/hsi) of University of Jyväskylä. 
You can use and modify this software under MIT licence. 

Currently, HyperBlend can only simulate point-spectrometer-like spectral data. It needs 
actual measured reflectance and transmittance data which it tries to replicate. 

## Installing

Clone the repository to some location on your machine. Create a python environment by running 
`conda env create -n hb --file hb_env.yml` in yor anaconda command prompt when in project root directory.
Use your favourite IDE for editing and running the code (developed using PyCharm). 
Command line build and run is untested, but it should work as well.

You will also need open-source 3D-modeling and rendering software Blender, which 
you can download and install from ([blender.org](blender.org)). At least versions 2.8x and 2.9x should 
work (developed on version 2.93.5). Change your Blender executable path to `constants.py`.

## Working principle

The measured reflectances and transmittances look like this:  

| wavelength [nm] | reflectance | transmittance
|---|---|---|
|400 | 0.21435 | 0.26547|
|401 | 0.21431 | 0.26540|
|... | ... | ... |

We call this the *target*. Reflectance and transmittance values represent the fraction of 
reflected and transmitted light so both values are separately bound to closed interval [0,1] 
and their sum cannot exceed 1. 

We use a Blender scene with a rectangular box that represents a leaf. The 
material of the leaf has four adjustable parameters: absorption particle density, scattering 
particle density, scattering anisotropy, and mix factor. These control how the light is scattered and 
absorbed in the leaf material.

For each wavelength in the target, we adjust the leaf material parameters until the modeled 
reflectance and transmittance match the target values.


## Usage

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
from src.leaf_model.optimization import Optimization
from data import toml_handlling as TH

data = [[400, 0.21435, 0.26547], [401, 0.21431, 0.26540]]
set_name = 'test_set'

o = Optimization(set_name)
TH.write_target(set_name, data, sample_id=0)
o.run_optimization()
```

The results are written onto disk in the set's directory as toml files and plotted to .png images.

## Project structure, *i.e.*, where to find stuff

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
