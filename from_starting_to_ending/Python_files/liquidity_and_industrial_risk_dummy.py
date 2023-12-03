import os
import pandas as pd
import numpy as np


file_path = (
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/"
)

df = pd.read_csv(
    os.path.join(file_path, "dependent_control_variable_filled_1.csv"), index_col=0
)
df = pd.DataFrame(df)
sensitive_data = pd.read_csv(os.path.join(file_path, "IND_RISK_DUMMY.csv"))
sensitive_data = pd.DataFrame(sensitive_data)
sensitive_data_0 = sensitive_data["ESG Prone Non-Sensitive Industries (0)"].to_list()

df["IND_RISK_DUMMY"] = 2
for companies in df["Company Name"].unique():
    if (
        companies + " IN Equity"
        in sensitive_data["ESG Prone Non-Sensitive Industries (0)"].to_list()
    ):
        df.loc[df["Company Name"] == companies, "IND_RISK_DUMMY"] = 0

    elif (
        companies + " IN Equity"
        in sensitive_data["ESG Prone Sensitive Industries (1)"].to_list()
    ):
        df.loc[df["Company Name"] == companies, "IND_RISK_DUMMY"] = 1

    else:
        df.loc[df["Company Name"] == companies, "IND_RISK_DUMMY"] = 0
        print(companies)
df = df.drop(["BS_CUR_ASSET_REPORT", "BS_CUR_LIAB"], axis=1)
df.to_csv(os.path.join(file_path, "dependent_control_variable_fully_filled.csv"))
