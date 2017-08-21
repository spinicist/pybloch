import numpy as np
from . import bloch_base as bb

def oneC(T2, tes):
        Mxy = np.exp(-tes / T2)
        return Mxy

def mc(T2s, fs, tes):
        Mxy = fs[0] * np.exp(-tes / T2s[0])
        for i in range(1, len(T2s)-1):
            Mxy = Mxy + fs[i] * np.exp(-tes / T2s[i])
        return Mxy