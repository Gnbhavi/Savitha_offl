import numpy as np
import pandas as pd

df = pd.read_csv(
    "~/Python_github/savitha/Savitha_offl/panel_data_predicting_esg_data__control_variable_filled.csv"
)
df = pd.DataFrame(df)

dropping_comp = pd.read_excel(
    "~/Python_github/savitha/Savitha_offl/Drop_list_90_firms.xlsx"
)
dropping_comp = pd.DataFrame(dropping_comp)
dropping_comp.columns = dropping_comp.iloc[0]
dropping_comp = dropping_comp.drop(dropping_comp.index[0])

dropping_comp_list = dropping_comp["Housing Finance Company"]
dropping_comp_list = dropping_comp_list[19:]

print(len(dropping_comp_list))
