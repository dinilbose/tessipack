#from distutils.core import setup
import codecs
from setuptools import setup
import os.path
import sys
sys.path.insert(0, "eleanor")

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")




with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
  name = 'tessipack',         # How you named your package folder (MyLib)
  packages = ['tessipack'],   # Chose the same as "name"
  version = get_version("tessipack/__init__.py"),      # Start with a small number and increase it with every change you make
  license='gpl-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Interactive package for analysing TESS FFI',   # Give a short description about your library
  author = 'Dinil Bose P',                   # Type in your name
  author_email = 'dinilbose@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/dinilbose/tessipack',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/dinilbose/ismrpy/archive/0.2.9.tar.gz',    # I explain this later on
  keywords = ['FFI', 'TESS'], # Keywords that define your package best
  long_description=long_description,
  package_data={'functions': ['README.md', 'LICENSE']},
  include_package_data=True,
  long_description_content_type="text/markdown",
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',   # Again, pick a license
    'Programming Language :: Python :: 3.6',
  ],
  install_requires=[
      'pandas>=0.13.1','bokeh==1.4.0',
      'photutils>=0.7', 'tqdm', 'lightkurve>=1.1.0', 'astropy',
      'astroquery','scipy',
      'setuptools>=41.0.0',
      'tensorflow<=1.14.0', 'vaneska', 'beautifulsoup4>=4.6.0', 'tess-point>=0.3.6'],
  setup_requires=["numpy","pandas"],
)
