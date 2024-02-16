# # Energy Companies
#
import os
from matplotlib import use
import numpy as np
import pandas as pd
from pandas.io.parsers.readers import fill
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_and_energy/energy/"
df1 = pd.read_csv(os.path.join(file_path, "Liquidity_added.csv"), index_col=0)


change_column = [
    "ESG_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
]
df1.loc[df1["Year"] == 2023, change_column] = 0

for val in change_column:
    df1[val] = df1[val].shift(1)

df1.drop(df1[df1["Year"] == 2003].index, inplace=True)

df1.to_csv(os.path.join(file_path, "time_lag.csv"))
