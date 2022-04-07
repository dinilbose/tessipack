Installation
=====
**tessipack** requires a python3.6 version with specific packages to run the program.
It would be beneficial for all users to create a virtual python environment to run the **tessipack**
package since it will not create any compatibility issues with other packages.
We urge users to create a virtual python environment to run the program.

.. _installation:

**Installation instruction for Ubuntu/Debian**

Python3.6
------------

To install python3.6 in your system if its not available.

.. code-block:: console

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6
    sudo apt install python3.6-venv


Virtual Environment
-------------------
To install virtual environment using pip

.. code-block:: console

    python3.6 -m pip install virtualenv


To create a directory where the virtual environment stores the data.

.. code-block:: console

    mkdir ~/python-virtual-environments && cd ~/python-virtual-environments
    python3.6 -m venv env

We prefer to create  ``python-virtual-environments`` folder in ``Home`` directory. *env* is the name of the virtual environment we have created. To activate the env environment.

.. code-block:: console

   source env/bin/activate

For deactivation of environment use (use this after the end of the installation)

.. code-block:: console

   deactivate

tessipack
---------

First lets update the pip package before proceeding for installation of tessipack.


.. code-block:: bash

   (env)$ pip install --upgrade pip
   (env)$ pip install tessipack







Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
