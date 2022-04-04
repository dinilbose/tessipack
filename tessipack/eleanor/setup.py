#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import setuptools
from setuptools import setup

sys.path.insert(0, "eleanor")
from version import __version__


long_description = \
    """
eleanor is a python package to extract target pixel files from
TESS Full Frame Images and produce systematics-corrected light curves
for any star observed by the TESS mission. In its simplest form, eleanor
takes a TIC ID, a Gaia source ID, or (RA, Dec) coordinates of a star
observed by TESS and returns, as a single object, a light curve and
accompanying target pixel data.
Read the documentation at https://adina.feinste.in/eleanor
Changes to v1.0.1 (2019-12-19):
* Ability to use local postcards
* Addition of eleanor.Update() for automatic sector updates
* Significant speedups when TIC, Coords, and a Gaia ID are all provided
* Other bug fixes

Changes to v1.0.0 (2020-01-14):
* Removed some package dependencies\
* Added clarifications in documentation
* Added new features to visualization tools
* Other bugfixes
"""



setup(
    name='eleanor',
    version=__version__,
    license='MIT',
    author='Adina D. Feinstein',
    author_email='adina.d.feinstein@gmail.com',
    packages=[
        'eleanor',
        ],
    include_package_data=True,
    url='http://github.com/afeinstein20/eleanor',
    description='Source Extraction for TESS Full Frame Images',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'functions': ['README.md', 'LICENSE']},
    install_requires=[
        'photutils>=0.7', 'tqdm', 'lightkurve>=1.1.0', 'astropy>=3.2.3',
        'astroquery', 'pandas',
        'setuptools>=41.0.0',
        'tensorflow<=1.14.0', 'vaneska', 'beautifulsoup4>=4.6.0', 'tess-point>=0.3.6'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.0',
        ],
    )
