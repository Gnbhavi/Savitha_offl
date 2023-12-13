import os
import pandas as pd


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/"
econ_df_1 = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_fully_filled.csv"), index_col=0
)

econ_df_1 = pd.DataFrame(econ_df_1)

independent_variables = [
    "ESG_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
]

for val in independent_variables:
    econ_df_1.loc[econ_df_1["Year"] == 2023, val] = 0
    econ_df_1[val] = econ_df_1[val].shift(1)


econ_df_1.to_csv(os.path.join(file_path, "timeLag_2003.csv"))
