import numpy as np
import pandas as pd

df = pd.read_csv(
    "~/Python_github/savitha/Savitha_offl/csv_files 9th_NOV/independent_dependent_control_variable_filled_except_levarage.csv"
)
df = pd.DataFrame(df)

dropping_comp = pd.read_excel(
    "~/Python_github/savitha/Savitha_offl/csv_files 9th_NOV/Drop_list_90_firms.xlsx"
)
dropping_comp = pd.DataFrame(dropping_comp)
dropping_comp.columns = dropping_comp.iloc[0]
dropping_comp = dropping_comp.drop(dropping_comp.index[0])

dropping_comp_list = dropping_comp["Housing Finance Company"]
dropping_comp_list = dropping_comp_list[19:].tolist()

df_after_dropping = df.copy()
for companies in dropping_comp_list:
    companyname = companies.replace(" IN Equity", "")
    df_after_dropping = df_after_dropping[
        df_after_dropping["Company Name"] != companyname
    ]


df_after_dropping.to_csv(
    "~/Python_github/savitha/Savitha_offl/csv_files 9th_NOV/independent_dependent_control_variable_filled_except_levarage_companies_dropped.csv"
)
