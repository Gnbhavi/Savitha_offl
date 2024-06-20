import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# The first chosen companies
# needed_companies = ["PAG","KPR", "RW", "TRID","ALOK", "WLSI", "ARVND", "VTEX", "KTG", "BD"]


needed_companies = ["PAG", "RW", "ALOK", "WLSI", "ARVND", "VTEX", "KTG", "BD"]

I_V = ['ESG_DISCLOSURE_SCORE', 'ENVIRON_DISCLOSURE_SCORE','SOCIAL_DISCLOSURE_SCORE', 'GOVNCE_DISCLOSURE_SCORE']

D_V = ['NET_WORTH_GROWTH', 'SALES_REV_TURN']

C_V = ['RETURN_ON_ASSET', 'IS_EPS', 'WACC']

needed_columns = ['Year', 'Company Name'] + I_V + D_V + C_V


df = pd.read_csv(  "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/panel_data_prediicting_2.csv",
    parse_dates=["Dates"],
)

# Remove the "Unnamed: 0" column (assuming it's the first column)
df = df.drop('Unnamed: 0', axis=1)  # Set axis=1 to drop columns

print(df.columns)

# Extract the year using the 'dt.year' attribute
df['Year'] = df['Dates'].dt.year


# Selecting the wanted companies
selected_companies_df = df.loc[df['Company Name'].isin(needed_companies)]

# Sorting and filtering needed coloumns
sorted_columns_df = selected_companies_df.reindex(columns=needed_columns)
filtered_df = sorted_columns_df.drop(sorted_columns_df.columns.difference(needed_columns), axis=1)

# year_filtered_df = filtered_df[filtered_df["Year"] >= 2012]

filtered_df.to_csv("/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_energy_and_Textile/textile/Dataset_creation/datset_only_with_needed_columns_and_rows.csv")

print("Job done")