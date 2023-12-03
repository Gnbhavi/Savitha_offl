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
val_count = 0
for companies in dropping_comp_list:
    companyname = companies.replace(" IN Equity", "")
    if companyname not in df["Company Name"].tolist():
        print(companyname)
        val_count += 1
    df_after_dropping = df_after_dropping[
        df_after_dropping["Company Name"] != companyname
    ]

print("Missing values: ", val_count)
print(len(df_after_dropping) / 21)
df_after_dropping.to_csv(
    "~/Python_github/savitha/Savitha_offl/csv_files 9th_NOV/independent_dependent_control_variable_filled_except_levarage_companies_dropped.csv"
)
