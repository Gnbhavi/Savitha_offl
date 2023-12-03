import os
import pickle
from re import subn
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np


def ESG_filler(data):
    valeus_1 = (
        data["ENVIRON_DISCLOSURE_SCORE"]
        + data["SOCIAL_DISCLOSURE_SCORE"]
        + data["GOVNCE_DISCLOSURE_SCORE"]
    )
    return valeus_1.round(4)


def other_independent_filler(data, known_variables):
    if len(known_variables) == 2:
        values_of = data[known_variables].sum(axis=1)
    else:
        values_of = data[known_variables]
    valeus_1 = (3 * data["ESG_DISCLOSURE_SCORE"] - values_of) / len(known_variables)
    return valeus_1.round(4)


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/independent_variables_wasted_companies_dict.pkl"

# open a file, where you stored the pickled data
file = open(file_path, "rb")

# dump information to that file
wasted_companies = pickle.load(file)

# close the file
file.close()


df = pd.read_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/independent_variables.csv"
)

df_copy = df.copy()
i = 0
for val in wasted_companies:
    if len(wasted_companies[val]) == 1 or len(wasted_companies[val]) == 2:
        i += 1
        print(f"{i} : {val}")
        data_value = df[df["Company Name"] == val]
        values = other_independent_filler(data_value, wasted_companies[val])
        for colmn in wasted_companies[val]:
            df_copy.loc[df_copy["Company Name"] == val, colmn] = values

df_copy.to_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/independent_variables_fully_filled.csv"
)
