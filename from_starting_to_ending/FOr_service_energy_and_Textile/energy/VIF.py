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


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_and_energy/energy/"
econ_df_1 = pd.read_csv(
    os.path.join(file_path, "dependent_and_control_filled.csv"),
    index_col=0,
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

corr = econ_df_1.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap="RdBu")
plt.show()

econ_df = econ_df_1.drop(["Company Name"], axis=1)
# set the data type and select rows up to 2016
econ_df = econ_df.astype(float)
econ_df_after = econ_df.drop(
    ["ENVIRON_DISCLOSURE_SCORE", "SOCIAL_DISCLOSURE_SCORE", "GOVNCE_DISCLOSURE_SCORE"],
    axis=1,
)


# the VFI does expect a constant term in the data, so we need to add one using the add_constant method
X1 = sm.tools.add_constant(econ_df)
X2 = sm.tools.add_constant(econ_df_after)
# X2 = sm.tools.add_constant(econ_df_after)
# create the series for both
series_before = pd.Series(
    [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])],
    index=X1.columns,
)
series_after = pd.Series(
    [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])],
    index=X2.columns,
)

# display the series
print("DATA BEFORE")
print("-" * 100)
print(series_before)
series_before.to_csv(os.path.join(file_path, "series_before.csv"))

print("DATA AFTER")
print("-" * 100)
print(series_after)
series_after.to_csv(os.path.join(file_path, "series_after.csv"))

desc_df = econ_df.describe()
desc_df = pd.DataFrame(desc_df)

desc_df.to_csv(os.path.join(file_path, "describe.csv"))
# add the standard deviation metric
desc_df.loc["+3_std"] = desc_df.loc["mean"] + (desc_df.loc["std"] * 3)
desc_df.loc["-3_std"] = desc_df.loc["mean"] - (desc_df.loc["std"] * 3)


# display it

pd.plotting.scatter_matrix(econ_df_after, alpha=1, figsize=(30, 20))
