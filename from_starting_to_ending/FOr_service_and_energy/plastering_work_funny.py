import os
import pandas as pd


file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/"
)
df = pd.read_csv(
    os.path.join(file_path, "new_files/dependent_control_variable_fully_filled.csv"),
    index_col=0,
)

df_1 = pd.read_csv(
    os.path.join(file_path, "FOr_service_and_energy/dependent_and_control_filled.csv"),
    index_col=0,
)

df_1.loc[df_1["Company Name"] == "SJET", "FNCL_LVRG"] = df.loc[
    df["Company Name"] == "SJET", "FNCL_LVRG"
]


df_1.loc[df_1["Company Name"] == "SJET", "PX_TO_BOOK_RATIO"] = df.loc[
    df["Company Name"] == "SJET", "PX_TO_BOOK_RATIO"
]

df_1["Liquidity"] = 0
for company in df_1["Company Name"].unique():
    df_1.loc[df_1["Company Name"] == company, "Liquidity"] = df.loc[
        df["Company Name"] == company, "Liquidity"
    ]


timelag_values = [
    "ESG_DISCLOSURE_SCORE",
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
]

for values_1 in timelag_values:
    df_1.loc[df_1["Year"] == 2023, values_1] = 0
    df_1[values_1] = df_1[values_1].shift(1)

df_1.to_csv(
    os.path.join(file_path, "FOr_service_and_energy/dependent_and_control_filled_3.csv")
)
