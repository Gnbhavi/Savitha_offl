import os
import numpy as np
import pandas as pd

file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/"
df = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_fully_filled.csv"), index_col=0
)
change_column = [
    "ESG_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
]
df_1 = df[df["Year"] > 2007]
df_1 = pd.DataFrame(df_1)
for colum_name in change_column:
    A_min = min(df_1[colum_name])
    df_1[colum_name] = df_1[colum_name].apply(
        lambda x: 2 * (1 + (x / A_min)) if x <= 0 else x
    )
    df_1[colum_name] = df_1[colum_name]

df_1 = df_1.drop(["Company Name", "Year"], axis=1)
corr_val = df_1.corr()

print(df_1.describe())
corr_val = pd.DataFrame(corr_val)
corr_val.to_csv(
    os.path.join(file_path, "correlation_laast15years_positive_changes.csv")
)

df_1.describe().round(4).to_csv(
    os.path.join(file_path, "describe_last15years_positive_changes.csv")
)
