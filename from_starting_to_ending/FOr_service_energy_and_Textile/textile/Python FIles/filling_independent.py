import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def ESG_filler(data):
    valeus_1 = (
        data["ENVIRON_DISCLOSURE_SCORE"]
        + data["SOCIAL_DISCLOSURE_SCORE"]
        + data["GOVNCE_DISCLOSURE_SCORE"]
    )
    return valeus_1.round(4)


def other_independent_filler(data, known_variables):
    valeus_1 = (3 * data["ESG_DISCLOSURE_SCORE"] - data["GOVNCE_DISCLOSURE_SCORE"]) * 0.45 
    valeus_2 = (3 * data["ESG_DISCLOSURE_SCORE"] - data["GOVNCE_DISCLOSURE_SCORE"]) * 0.55
    
    return valeus_1.round(4), valeus_2.round(4)


I_V = ['ESG_DISCLOSURE_SCORE', 'ENVIRON_DISCLOSURE_SCORE','SOCIAL_DISCLOSURE_SCORE', 'GOVNCE_DISCLOSURE_SCORE']

D_V = ['NET_WORTH_GROWTH', 'SALES_REV_TURN']

C_V = ['RETURN_ON_ASSET', 'IS_EPS', 'WACC']


df = pd.read_csv( "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/datset_only_with_needed_columns_and_rows.csv",
    parse_dates=["Year"],
    index_col="Year",
)

df =df.drop(["Unnamed: 0"], axis=1)

wasted_companies = {}

df_copy = df.copy()

for companies in df["Company Name"].unique():
    wasted_companies[companies] = []
    for variables in I_V:
        df_copy_company = df[df["Company Name"] == companies]
        variable_col = df_copy_company[variables]
        variable_col_1 = variable_col.dropna()
        if len(variable_col_1) == 0:
            wasted_companies[companies].append(variables)
            continue
        if len(variable_col_1) == len(variable_col):
            continue

        first_non_nan_index = variable_col.first_valid_index()
        first_non_nan_index = variable_col.index.get_loc(first_non_nan_index)
        print(first_non_nan_index)

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
       

# df_copy.to_csv("/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/useless_files_dont_mind.csv")


# for company_name in wasted_companies:
#     if len(wasted_companies[company_name]) == 4:
#         df_copy = df_copy[df_copy["Company Name"] != company_name]

i = 0
for val in wasted_companies:
    if len(wasted_companies[val]) == 1 or len(wasted_companies[val]) == 2:
        i += 1
        print(f"{i} : {val}")
        data_value = df_copy[df_copy["Company Name"] == val]
        # values = other_independent_filler(data_value, wasted_companies[val])
        df_copy.loc[df_copy["Company Name"] == val, 'ENVIRON_DISCLOSURE_SCORE'] = 0
        df_copy.loc[df_copy["Company Name"] == val, 'SOCIAL_DISCLOSURE_SCORE'] = 0


# df_copy.to_csv("/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/useless_files_dont_mind_2.csv")

print(df_copy)