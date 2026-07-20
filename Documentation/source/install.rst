.. _chap-install:

Installing HyperBlend
==============================

There is not a real installation process for HyperBlend.
You can just clone the repository and run it with your favourite IDE or using a console.
For running the simulations, you will also need Blender 3.6.X.
We look at how to get everything working on Windows or Linux. MacOS is not supported.


Cloning the project
---------------------------

This is very simple: start by creating a directory for the project, open
a terminal to that directory and use command

.. code-block::

    git clone <the.address.from.where.to.clone.from>

to clone the Github repository of HyperBend in that location.
Let's assume that is now `<your path>/HyperBlend/` and call it the project **root**.
You can now find the source code from `root/src/`, for example.


Setting up Conda environment
------------------------------

HyperBlend relies on a bunch of external libraries that are specified in
`root/hb_env.yml` file. To install these, you should use a Python package
handler such as  MiniConda `anaconda.com <https://www.anaconda.com>`__
or Mamba `github.com/mamba-org/mamba <https://github.com/mamba-org/mamba>`__.
Both are free to use but Mamba may be faster to run in many cases.

You can test that your installation was successful by running

.. code-block::

    conda --version

or

.. code-block::

    mamba --version

in your command prompt. (For Windows, this would be the Anaconda command prompt, not
the default one and not Powershell.)

Once that is done, run:

.. code-block::

    conda env create -n hb --file hb_env.yml

which will create a new virtual Python environment called `hb`. Naturally,
replace word `conda` with `mamba` if you are using mamba.


Installing Blender
--------------------

One final thing before running HyperBlend is the installation of
Blender 3D-modeling and rendering software. Blender is open-source
software available at https://www.blender.org/. HyperBlend works only with
Blender version 3.6.XX, which you can download from
`download.blender.org/release/Blender3.6 <https://download.blender.org/release/Blender3.6/>`__.
There are installers for Windows (`windows-x64.msi`) and
Linux (`linux-x64.tar.xz`).

.. note::
    You have to install Blender 3.6 even if you have another
    version already installed if you want to run any simulations.

Follow the installation wizard and take note on where you installed this
specific Blender version as we need that path next.

HyperBlend needs to know which Blender to call. For that reason,
open file `root/src/constants.py` (with your favourite text editor,
for example) and depending on your operating system, replace the content
of variable `blender_executable_path_win` or 'blender_executable_path_linux'
with your installation path. On Windows, HyperBlend tries to find the correct
Blender version automatically, so it may work even without the specified path.

What then?
------------

The rest of the documentation assumes you have some sort of IDE to run
Python software. The examples assume PyCharm but any will do. Of course,
you can also run Python from command line, but you have to figure out how
to do that by yourself.

Now we are ready to test run HyperBlend. Move to the
:doc:`beginner tutorial <./basics>`.
