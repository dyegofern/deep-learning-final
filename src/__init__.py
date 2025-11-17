"""
GAN-Based Carbon Emissions Prediction for Aviation
CSCA 5642 - Final Project
University of Colorado Boulder

Source code package for aviation emissions prediction using CTGAN.
"""

__version__ = '1.0.0'
__author__ = 'CSCA 5642 Student'

from . import data_processing
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = [
    'data_processing',
    'models',
    'training',
    'evaluation',
    'utils'
]
