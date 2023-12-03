import pandas as pd
import numpy as np
from scipy.sparse.linalg import use_solver


df = pd.read_csv(
    "~/Python_github/savitha/Savitha_offl/Nov_19_23/env_dis_scr_filled.csv"
)
df = pd.DataFrame(df)
df = df.drop(["Unnamed: 0"], axis=1)
# df["Dates"] = pd.to_datetime(df["Dates"], format="%y/%m/%d")
# df["Year only"] = df["Dates"].dt.year
# fill_col = "ENVIRON_DISCLOSURE_SCORE"
# use_col = ["ESG_DISCLOSURE_SCORE", "SOCIAL_DISCLOSURE_SCORE", "GOVNCE_DISCLOSURE_SCORE"]
#
neg_remove_col = [
    "ESG_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
]

# Fill null values in the specified column with the sum of nearby columns
# df[fill_col] = df.apply(
#     lambda row: row[fill_col]
#     if pd.notna(row[fill_col])
#     else 3 * row[use_col[0]] - row[use_col[1]] - row[use_col[2]],
#     axis=1,
# )
# print(df["ENVIRON_DISCLOSURE_SCORE"].count())
# exit()
# df.to_csv("~/Python_github/savitha/Savitha_offl/Nov_19_23/env_dis_scr_filled.csv")
df_last_15 = df[df["Year only"] > 2007]

df_1 = df_last_15.copy()
for column_name in neg_remove_col:
    min_val_col = min(df_1[column_name])
    df_1[column_name] = df_1[column_name].apply(
        lambda x: 2 * (1 + (x / (int(min_val_col) + 2))) if x <= 0 else x
    )
# df_1["SOCIAL_DISCLOSURE_SCORE"] = df_1["SOCIAL_DISCLOSURE_SCORE"].apply(
#     lambda x: 2 * (1 + (x / 89)) if x <= 0 else x
# )
describe_values = df_1.describe()
describe_values = pd.DataFrame(describe_values)

describe_values.to_csv(
    "~/Python_github/savitha/Savitha_offl/Nov_19_23/from_year_2008_desribe_1.csv"
)
