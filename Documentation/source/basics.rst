
Beginner Tutorial
=======================

In the :doc:`previous step <./install>`, we installed everything needed
to run HyperBlend, so it's time to test it out.

First Run
----------

HyperBlend does not have any kind of user interface, so you will have to
do a bit of programming to run it. If you go to the entrypoint of the
program in `root/src/__main__.py` there should only be one method which
looks something like this

.. code-block:: python3

    if __name__ == "__main__":

        runtime = initialization.initialize()

You can comment out everything there possibly is after this line and
just run the initialization to verify that everything is in order.
You can check the documentation of the initialization method from
:py:func:`setup.initialization.initialize`.

In Case of Errors
-----------------

If the Blender path is not correct or the Blender version is a wrong one
(the version is only checked on Windows), you will get an error. Check
that you installed the correct version and that you provided a correct path.

Another thing that is likely to occur is that you get an import error from
every script in the program.

The imports in HyperBlend require `<wherever_you_cloned_the_repo>/HyperBlend/`, i.e.,
the `root`, as we decided to call it in the installation tutorial,
directory to be included in `PYTHONPATH` variable.
You can check if it is by running

.. code-block:: python3

    import sys
    print(sys.path)

This will print you a list of paths in the variable and one of the should be the
`root`. If not, and if you are using PyCharm as your IDE on Windows machine, you can

- go to `<wherever_you_cloned_the_repo>/`
- right-click on HyperBlend directory
- select "Open as PyCharm project"

The IDE will then know that  `<wherever_you_cloned_the_repo>/HyperBlend/` should
be added to `PYTHONPATH`.

You can also set the `PYTHONPATH` system wide manually for any OS following
the instructions in
https://www.simplilearn.com/tutorials/python-tutorial/python-path, for example.


HyperBlend Simulations
-----------------------

This subsection contains a little introduction to HyperBlend's working
principle necessary to understand the bare basics. We will revisit these
ideas in more depth later. If you are already too eager to run some code,
you can skip to the next heading and copypaste run it, and come back here
after that.

HyperBlend is designed for creating spectral forest canopy simulations.
Since the leaves of trees are the most active part interacting with light
the first part of the simulation is focused on that. In this tutorial,
this is called with a general term :term:`Slab simulation`. Creating
custom slab simulations is explained in more detail in
:ref:`chap-slab-simulation` chapter.



The Smallest Working Example
-----------------------------

To be filled in very soon.