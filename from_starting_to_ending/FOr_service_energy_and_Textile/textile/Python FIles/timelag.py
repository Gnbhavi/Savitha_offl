import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


I_V = ['ESG_DISCLOSURE_SCORE', 'ENVIRON_DISCLOSURE_SCORE','SOCIAL_DISCLOSURE_SCORE', 'GOVNCE_DISCLOSURE_SCORE']

df = pd.read_csv( "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/Fully_filled_data_ready_for_use.csv",
    parse_dates=["Year"],
)
df["Year"] = df['Year'].dt.year

for values in I_V:
  df[values] = df[values].shift(1)

filtered_year_df = df[df['Year'] >= 2015]

filtered_year_df.to_csv("/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/year_filtered_ready_for_model_training.csv")