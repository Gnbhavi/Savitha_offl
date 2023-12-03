import os
import numpy as np
import pandas as pd
from pandas.core.api import DataFrame


file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/"
)
df_filler = pd.read_csv(os.path.join(file_path, "test_file.csv"), index_col=0)

df = pd.read_csv(
    os.path.join(file_path, "independent_variables_fully_filled.csv"), index_col=0
)

df_1 = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_fully_filled.csv"), index_col=0
)

df_2 = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_filled_1.csv"), index_col=0
)
# A = np.where(df["SOCIAL_DISCLOSURE_SCORE"].isnull())[0]
#
# df["SOCIAL_DISCLOSURE_SCORE"][A].value = df["ESG_DISCLOSURE_SCORE"][A] / 3


df_1["SOCIAL_DISCLOSURE_SCORE"] = df_1["SOCIAL_DISCLOSURE_SCORE"].fillna(
    df["ESG_DISCLOSURE_SCORE"] / 3
)
df_2["SOCIAL_DISCLOSURE_SCORE"] = df_1["SOCIAL_DISCLOSURE_SCORE"].fillna(
    df["ESG_DISCLOSURE_SCORE"] / 3
)
# B = np.where(df_1["SOCIAL_DISCLOSURE_SCORE"].isnull())[0]
# df_1["SOCIAL_DISCLOSURE_SCORE"][B].value = df_1["ESG_DISCLOSURE_SCORE"][B] / 3
#
# C = np.where(df_2["SOCIAL_DISCLOSURE_SCORE"].isnull())[0]
# df_2["SOCIAL_DISCLOSURE_SCORE"][C].value = df_2["ESG_DISCLOSURE_SCORE"][C] / 3

# df.to_csv(
#     os.path.join(file_path, "independent_variables_fully_filled_social_filled.csv")
# )
df_1.to_csv(
    os.path.join(file_path, "new_files/dependent_control_variable_fully_filled.csv")
)
df_2.to_csv(
    os.path.join(file_path, "new_files/dependent_control_variable_filled_1.csv")
)

df_1.describe().to_csv(os.path.join(file_path, "social_filled_fullly.csv"))
exit()
df = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_filled_1.csv"), index_col=0
)
df["Dates"] = pd.to_datetime(df["Dates"], format="%d-%m-%Y")
df["Year"] = df["Dates"].dt.year
df = df[["Year"] + df.columns.tolist()[:-1]]
df = df.drop("Dates", axis=1)

df.loc[(df["Company Name"] == "BCG") & (df["Year"] == 2009), "BS_CUR_LIAB"] = 130.63
df.loc[(df["Company Name"] == "BCG") & (df["Year"] == 2010), "BS_CUR_LIAB"] = 160.69

print(
    df.loc[(df["Company Name"] == "BCG") & (df["Year"] == 2009), "BS_CUR_LIAB"].values
)

df["Liquidity"] = df["BS_CUR_ASSET_REPORT"] / df["BS_CUR_LIAB"]
df.to_csv(os.path.join(file_path, "dependent_control_variable_filled_1.csv"))
