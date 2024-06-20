# Service Companies

import os
from matplotlib import use
import numpy as np
import pandas as pd
from pandas.io.parsers.readers import fill
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending"
)
df = pd.read_csv(
    os.path.join(file_path, "independent_variables_fully_filled_social_filled.csv"),
    index_col=0,
)


# Convert the 'Birthdate' column to datetime
df["Dates"] = pd.to_datetime(df["Dates"])

# Extract the year and create a new 'Year' column
df["Year"] = df["Dates"].dt.year
service_companies_list = [
    "INDIGO",
    "SJET",
    "CCRI",
    "BDE",
    "AGLL",
    "TRPC",
    "VRLL",
    "GRFL",
    "RIIL",
    "GESCO",
    "SCI",
    "GMRI",
    "ADSEZ",
    "GPPV",
    "BFUT",
    "REDI",
    "MMTC",
    "SECIS",
    "NSE",
    "FSOL",
    "ECLX",
    "HGSL",
]


change_column = [
    "ESG_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
]
filling_column = [
    "FNCL_LVRG",
    "RETURN_ON_ASSET",
    "BS_TOT_ASSET",
    "PX_TO_BOOK_RATIO",
]
# "Liquidity",
useful_column = ["Year", "Company Name"] + change_column + filling_column
print(useful_column)
# df_1 = df[df["Year"] > 2007]
df = df[df["Company Name"].isin(service_companies_list)]
df_copy = df.copy()
df_copy = pd.DataFrame(df_copy)
for companies in service_companies_list:
    for variables in filling_column:
        df_copy_company = df[df["Company Name"] == companies]
        variable_col = df_copy_company[variables]
        variable_col_1 = variable_col.dropna()
        if len(variable_col_1) == 0:
            print("Company name: ", companies, "column: ", variables)
            continue
        if len(variable_col_1) == len(variable_col):
            continue

        first_non_nan_index = variable_col.first_valid_index()
        first_non_nan_index = variable_col.index.get_loc(first_non_nan_index)

        last_non_nan_index = variable_col.last_valid_index()
        last_non_nan_index = variable_col.index.get_loc(last_non_nan_index)
        nan_indices = np.where(np.isnan(variable_col.values))[0]

        decom = sm.tsa.seasonal_decompose(variable_col_1, model="additive", period=1)
        trend_component = decom.trend
        # Assuming you have a time index for your data
        X = range(len(trend_component))
        X = sm.add_constant(X)  # Add a constant for the intercept
        y = trend_component

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the remaining values in the trend\
        nan_indices_1 = nan_indices.copy()
        for i in range(len(nan_indices)):
            if nan_indices[i] < first_non_nan_index:
                nan_indices_1[i] = nan_indices[i] - first_non_nan_index
            else:
                nan_indices_1[i] = nan_indices[i] - last_non_nan_index

        if len(nan_indices_1) == 1:
            predicted_trend = model.predict([np.append(1, nan_indices_1)])
        else:
            previous_X = sm.add_constant(nan_indices_1)
            predicted_trend = model.predict(previous_X)

        predicted_trend = dict(zip(nan_indices, predicted_trend))

        variable_col2 = variable_col.to_list()
        K = [
            predicted_trend[i] if i in nan_indices else variable_col2[i]
            for i in range(len(variable_col))
        ]

        K = np.round(K, 4)

        df_copy.loc[df_copy["Company Name"] == companies, variables] = K

df_copy = df_copy[useful_column]
df_copy["SOCIAL_DISCLOSURE_SCORE"] = np.round(df["SOCIAL_DISCLOSURE_SCORE"], 4)
df_copy["Firm size"] = np.log(np.abs(df_copy["BS_TOT_ASSET"]))
main_file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_and_energy/"
df_copy.to_csv(os.path.join(main_file_path, "dependent_and_control_filled.csv"))
