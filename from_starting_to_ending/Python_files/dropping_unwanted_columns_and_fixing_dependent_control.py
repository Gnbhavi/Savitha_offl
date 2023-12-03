import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending"
)
df_1 = pd.read_csv(os.path.join(file_path, "dependent_control_variable_filled_1.csv"))
# finc_comp_drop = pd.read_csv(os.path.join(file_path, "Finannce_companies_drop.csv"))
# word_to_remove = " IN Equity"
# j = 0
# dropping_comp_val = []
# for companies in finc_comp_drop["Company"]:
#     companies_value = companies.replace(word_to_remove, "")
#     if companies_value in df_1["Company Name"].unique():
#         j += 1
#         print(f"{j}) {companies_value}")
#         dropping_comp_val.append(companies_value)
#         df_1 = df_1[df_1["Company Name"] != companies_value]
# dropping_comp_val = pd.DataFrame(dropping_comp_val)
# dropping_comp_val.to_csv(os.path.join(file_path, "finance_drop_83.csv"))

Needed_variables = [
    "Dates",
    "Company Name",
    "ESG_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
    "ALTMAN_Z_SCORE",
    "TOBIN_Q_RATIO",
    "RETURN_ON_ASSET",
    "FNCL_LVRG",
    "BOOK_VAL_PER_SH",
    "PX_TO_BOOK_RATIO",
    "NET_WORTH_GROWTH",
    "BS_CUR_ASSET_REPORT",
    "BS_CUR_LIAB",
    "TOT_DEBT_TO_COM_EQY",
]

dependent_variables = [
    "ALTMAN_Z_SCORE",
    "TOBIN_Q_RATIO",
    "RETURN_ON_ASSET",
]

control_variable = [
    "TOT_DEBT_TO_COM_EQY",
    "FNCL_LVRG",
    "BOOK_VAL_PER_SH",
    "PX_TO_BOOK_RATIO",
    "NET_WORTH_GROWTH",
]

dropping_column = ["BS_CUR_ASSET_REPORT", "BS_CUR_LIAB"]

# df_needed_only = df_1[Needed_variables]


variable_to_be_filled = dependent_variables + control_variable + dropping_column
wasted_companies = {}

df_copy = df_1.copy()
companies = "SJET"
wasted_companies[companies] = []
for variables in variable_to_be_filled:
    df_copy_company = df_copy[df_copy["Company Name"] == companies]
    variable_col = df_copy_company[variables]
    variable_col_1 = variable_col.dropna()
    if len(variable_col_1) == 0:
        wasted_companies[companies].append(variables)
        print(variable_col_1)
        print(companies)
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

# df_copy["Liquidity"] = df_copy[dropping_column[0]] / df_copy[dropping_column[1]]
# df_copy = df_copy.drop(dropping_column, axis=1)

df_copy = df_copy.drop("Unnamed: 0", axis=1)
print(len(df_copy.columns))
df_copy.to_csv(os.path.join(file_path, "dependent_control_variable_filled_1.csv"))

# with open(
#     os.path.join(file_path, "independent_variables_wasted_companies_dict.pkl"),
#     "wb",
# ) as pickle_file:
#     pickle.dump(wasted_companies, pickle_file)
