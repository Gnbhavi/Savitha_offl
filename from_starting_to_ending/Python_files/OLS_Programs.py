import os
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/"
econ_df_1 = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_fully_filled.csv"), index_col=0
)
econ_df_1 = econ_df_1[econ_df_1["Year"] > 2007]
econ_df_1 = pd.DataFrame(econ_df_1)
# set the index to the year column
econ_df_1 = econ_df_1.set_index("Year")
for column_name in econ_df_1.columns:
    if econ_df_1[column_name].isnull().values.any():
        print(column_name)
        index = econ_df_1[column_name].index[econ_df_1[column_name].apply(np.isnan)]
        # inds = pd.isnull(econ_df_1).any(column_name).nonzero()[0]
        df_index = econ_df_1.index.values.tolist()
        inds = [df_index.index(i) for i in index]
        print(inds)
econ_df = econ_df_1.drop(["Company Name"], axis=1)
# set the data type and select rows up to 2016
econ_df = econ_df.astype(float)


# the VFI does expect a constant term in the data, so we need to add one using the add_constant method
X1 = sm.tools.add_constant(econ_df)
# X2 = sm.tools.add_constant(econ_df_after)
# create the series for both
series_before = pd.Series(
    [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])],
    index=X1.columns,
)
# series_after = pd.Series([variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])], index=X2.columns)

# display the series
print("DATA BEFORE")
print("-" * 100)
print(series_before)
exit()


print("DATA AFTER")
print("-" * 100)
display(series_after)
# calculate the correlation matrix
corr = econ_df.corr()
corr = abs(corr)

# display the correlation matrix
corr.to_csv(
    "~/Python_github/savitha/Savitha_offl/21_NOV_2023/correlation_matrix_absolute.csv"
)

# plot the correlation heatmap
hm = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="RdBu")

plt.show()
