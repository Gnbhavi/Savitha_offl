import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.nanops import get_corr_func


class CorrelationsWithPath:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)  # For Reading CSV file

    def get_correlations(self, columns, storing_location):
        corr = self.df[columns].corr()  # For fiding correlation between columns
        corr.to_csv(storing_location)  # Stores the correlation in the given location
        return corr


df = pd.read_excel(
    "~/Python_github/savitha/Savitha_offl/15th Nov 2023/independent_dependent_control_variable_filled_92_dropped_Final_Excel.xlsx"
)
df = df.drop(["Unnamed: 0"], axis=1)
df = pd.DataFrame(df)
columns_values = df.columns.tolist()
columns_values = columns_values[2:]
corr1 = df[columns_values].corr()
print(corr1)
corr1.to_csv("~/Python_github/savitha/Savitha_offl/15th Nov 2023/correlation.csv")
