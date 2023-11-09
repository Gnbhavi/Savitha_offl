import json
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/gnbhavithran/Python_github/savitha/02_09_23/panel_data_prediicting_2.csv", parse_dates=['Dates'], index_col='Dates')

df = df.drop([ "Unnamed: 0"], axis=1)

df = df.drop(['IS_ADVERTISING_EXPENSES', 'IS_TOTAL_REVENUES', 'ARDR_NI_TO_SE_RATIO'], axis=1)

# df = df[df['Company Name'] == "SCHN"]

variable_to_be_filled = ['ENVIRON_DISCLOSURE_SCORE', 'ESG_DISCLOSURE_SCORE', 'SOCIAL_DISCLOSURE_SCORE', 'GOVNCE_DISCLOSURE_SCORE']

# variable_to_be_filled = ['ALTMAN_Z_SCORE', 'TOBIN_Q_RATIO', 'RETURN_ON_ASSET', 'RETURN_COM_EQY', 'SALES_REV_TURN', 'IS_EPS', 'FNCL_LVRG', 'BOOK_VAL_PER_SH', 'PX_TO_BOOK_RATIO', 'NET_WORTH_GROWTH', 'TOT_DEBT_TO_COM_EQY', 'BS_CUR_ASSET_REPORT', 'BS_CUR_LIAB']

# variable_to_be_filled = 'ALTMAN_Z_SCORE'

df_copy = df.copy()
wasted_companies = []
# useful_column = {}
# wasted_column ={}
# wasted_column2 = {}

with open('4-2-1.pkl', 'rb') as f:
    useful_column = pickle.load(f)


for companies in df['Company Name'].unique():
    # wasted_column[companies] = []
    for variables in variable_to_be_filled:

        df_copy_company = df[df['Company Name'] == companies]
        variable_col = df_copy_company[variables]
        variable_col_1 = variable_col.dropna()
        if len(variable_col_1) == 0:
            wasted_companies.append(companies)
            # wasted_column[companies].append(variables)
            continue
        if len(variable_col_1) == len(variable_col):
            continue

        first_non_nan_index = variable_col.first_valid_index()
        first_non_nan_index = variable_col.index.get_loc(first_non_nan_index)

        last_non_nan_index = variable_col.last_valid_index()
        last_non_nan_index = variable_col.index.get_loc(last_non_nan_index)
        nan_indices = np.where(np.isnan(variable_col.values))[0]

        decom = sm.tsa.seasonal_decompose(variable_col_1, model='additive', period=1)
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
        K = [predicted_trend[i] if i in nan_indices else variable_col2[i] for i in range(len(variable_col))]
        # K = np.append(predicted_trend, variable_col)
        K = np.round(K, 4)
        df_copy.loc[df_copy['Company Name'] == companies, variables] = K
    # if len(wasted_column[companies]) >= 1 and len(wasted_column[companies]) < 4:
    #   wasted_column2[companies] = wasted_column[companies]

# print(len(wasted_column2))

# a_file = open("data.json", "w")
# json.dump(wasted_column2, a_file)

# with open('4-2-1.pkl', 'wb') as f:
#     pickle.dump(wasted_column2, f)

# exit()

for company_name in set(wasted_companies):
    if company_name not in useful_column.keys():
        df_copy = df_copy[df_copy['Company Name'] != company_name]


useful_companies = df_copy['Company Name'].unique()
# # np.savetxt("Useful Companies.txt", useful_companies)
useful_companies = pd.DataFrame(useful_companies)
useful_companies.to_csv("useful_companies.csv")

# print(wasted_column)
df_copy.to_csv("panel_data_predicting_esg_data_filled_1.csv")
print("Done")
