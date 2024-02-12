# Energy Companies

import os
from matplotlib import use
import numpy as np
import pandas as pd
from pandas.io.parsers.readers import fill
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/"
)
df = pd.read_csv(
    os.path.join(
        file_path, "FOr_service_and_energy/energy/dependent_and_control_filled.csv"
    ),
    index_col=0,
)
