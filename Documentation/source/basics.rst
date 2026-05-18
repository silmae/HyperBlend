
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

The imports in HyperBlend require `<wherever_you_cloned_the_repo>/HyperBlend/`
directory to be included in `PYTHONPATH` variable.
You can check if it is by running

.. code-block:: python3

    import sys
    print(sys.path)

If not, and if you are using PyCharm as your IDE on Windows machine, you can

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

A tiny introduction to the working principle of HB.
See the next heading if you want to try running a tiny example.

The Smallest Working Example
-----------------------------

To be filled in very soon.