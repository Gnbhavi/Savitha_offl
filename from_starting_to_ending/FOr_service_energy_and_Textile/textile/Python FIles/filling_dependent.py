import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


I_V = ['ESG_DISCLOSURE_SCORE', 'ENVIRON_DISCLOSURE_SCORE','SOCIAL_DISCLOSURE_SCORE', 'GOVNCE_DISCLOSURE_SCORE']

D_V = ['NET_WORTH_GROWTH', 'SALES_REV_TURN']

C_V = ['RETURN_ON_ASSET', 'IS_EPS', 'WACC']


df = pd.read_csv( "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/useless_files_dont_mind_2.csv",
    parse_dates=["Year"],
    index_col="Year",
)

variable_to_be_filled = D_V + C_V


# df =df.drop(["Unnamed: 0"], axis=1)

df_copy = df.copy()
wasted_companies = {}

for companies in df_copy["Company Name"].unique():
  wasted_companies[companies] = []
  df_copy_company = df[df["Company Name"] == companies]
  for variables in variable_to_be_filled:
      variable_col = df_copy_company[variables]
      variable_col_1 = variable_col.dropna()
      if len(variable_col_1) == 0:
          print(variable_col_1)
          wasted_companies[companies].append(variable_col_1)
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

df_copy.to_csv("/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/Fully_filled_data_ready_for_use.csv")

# df_copy["Liquidity"] = df_copy[dropping_column[0]] / df_copy[dropping_column[1]]
# df_copy = df_copy.drop(dropping_column, axis=1)

# df_copy = df_copy.drop("Unnamed: 0", axis=1)
# df_copy.to_csv(os.path.join(file_path, "dependent_control_variable_filled_1.csv"))
