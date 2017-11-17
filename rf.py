#!/usr/bin/env python
"""rf.py

RF Pulse Objects"""
import numpy as np
from scipy import special

# pylint rules on var / arg names and numbers do not suit science
# pylint: disable=C0103,R0913,R0914

_gamma = 42.6e6 # Hz per Tesla
_gamma2pi = _gamma * 2 * np.pi # Radians per Tesla
_defaultB0 = 3

class RF:
    def __str__(self):
        return type(self).__name__ + ', '.join("%s: %s" % item for item in vars(self).items())

class Hard(RF):
    """Basic hard/block RF pulse"""
    def __init__(self, flip_d, Trf, B0 = _defaultB0):
        flip_r = np.radians(flip_d)
        self.flip_d = flip_d
        self.flip_r = flip_r
        self.length = Trf
        # For hard pulses, the below is a bit pointles...
        self.amp = flip_r / (B0 * _gamma2pi * Trf)
        self.int_omega2 = Trf * (self.amp * B0 * _gamma2pi)**2

class Sinc(RF):
    """Sinc RF pulse"""
    def __init__(self, flip_d, Trf, side_lobes = 1, B0 = _defaultB0):
        flip_r = np.radians(flip_d)
        self.flip_d = flip_d
        self.flip_r = flip_r
        self.length = Trf
        self.N = 2 * (side_lobes + 1) # +1 for central lobe, *2 for symmetric
        Npi = self.N * np.pi
        self.amp = flip_r / (B0 * _gamma2pi * Trf * special.sici(Npi)[0] / Npi)
        self.int_omega2 = Trf * (B0 * _gamma2pi * self.amp)**2 * special.sici(2*Npi)[0] / Npi

class Rui(RF):
    """Rui's special pulses"""
    def __init__(self, flip_d, Trf, side_lobes = 1, target_flip = 90, B0 = _defaultB0):
        self.flip_d = flip_d
        self.flip_r = np.radians(flip_d)
        self.length = Trf
        target = Sinc(target_flip, Trf, side_lobes)
        self.int_omega2 = target.int_omega2