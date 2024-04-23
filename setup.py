from setuptools import setup, Extension
import os
from revpref import __version__, __author__

setup(
    name = 'revpref',
    version = __version__ ,
    author = __author__,
    url = 'https://github.com/Aliebc/revpref',
    description = 'Python Tools for Computational Revealed Preference Analysis',
    packages = ['revpref'],
    install_requires = [
        'numpy', 
        'networkx', 
        'pulp>=2.6.0', 
        'scipy>1.10', 
        'matplotlib'
    ],
)