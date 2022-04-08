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

To install python3.6 in your system if it is not available.

.. code-block:: console

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.6
    sudo apt install python3.6-venv
    sudo apt install python3-pip
    sudo apt-get install python3-venv



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

First, let's update the pip package before proceeding with the installation of the **tessipack**.


.. code-block:: bash

   (env)$ pip install --upgrade pip
   (env)$ pip install tessipack


GUI
---
The graphical interface of **tessipack** runs with **bokeh** package.
``bokeh serve`` command is used to run the  GUI.
To run the GUI we have to locate the location of the installation directory of **tessipack** package.
Since we use the virtual environment the command is as follows.

.. code-block:: bash

  (env)$ python3.6 -m bokeh serve /home/dinilbose/python-virtual-environments/env/lib/python3.6/site-packages/tessipack/gui/

We can create an easy run script for running the GUI. An example of run ``runtessipack`` is as follows.

.. code-block:: bash

   #!/bin/bash
   source ~/python-virtual-environments/env/bin/activate
   python3.6 -m bokeh serve /home/dinilbose/python-virtual-environments/env/lib/python3.6/site-packages/tessipack/gui/

make the run script executable via this command

.. code-block:: bash

   chmod +x runtessipack

Run the  program using


.. code-block:: bash

   ./runtessipack
