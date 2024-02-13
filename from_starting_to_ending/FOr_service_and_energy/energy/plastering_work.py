# # Energy Companies
#
import os
from matplotlib import use
import numpy as np
import pandas as pd
from pandas.io.parsers.readers import fill
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/"
)
df1 = pd.read_csv(
    os.path.join(
        file_path, "FOr_service_and_energy/energy/dependent_and_control_filled.csv"
    ),
    index_col=0,
)

df2 = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_fully_filled.csv"), index_col=0
)
print(df1.columns)
print(df2.columns)
columns = list(df1.columns) + ["Liquidity"]
merged_df = pd.merge(
    df1,
    df2[["Year", "Company Name", "Liquidity"]],
    on=["Year", "Company Name"],
    how="left",
)
# merged_df = merged_df[columns]

merged_df.to_csv(
    os.path.join(file_path, "FOr_service_and_energy/energy/Liquidity_added.csv")
)
# import pandas as pd
#
# # Sample DataFrames
# df1 = pd.DataFrame(
#     {
#         "Year": [2020, 2020, 2021, 2021, 2022, 2022],
#         "Company": ["A", "B", "C", "A", "B", "C"],
#     }
# )
#
# df2 = pd.DataFrame(
#     {
#         "Year": [2020, 2021, 2022, 2020, 2021, 2022],
#         "Company": ["A", "B", "C", "D", "E", "F"],
#         "New_Column_Data": ["data1", "data2", "data3", "data4", "data5", "data6"],
#     }
# )
#
# # Merge the two DataFrames on 'Year' and 'Company' with left join
# merged_df = pd.merge(df1, df2, on=["Year", "Company"], how="left")
#
# # Select only the columns from df1
# # merged_df = merged_df[df1.columns]
# #
# # Display the merged DataFrame
# print(merged_df)
