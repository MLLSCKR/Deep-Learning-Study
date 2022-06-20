"""
Date 220616.
    MLP 직접 구현해 보기 with using class. (by SCL)
        VER01(220616. only for regression)
"""

# mathutil

import pandas as pd
import numpy as np
import os

def relu(x):
    return np.maximum(x, 0)

def derv_relu(x):
    return np.sign(x)