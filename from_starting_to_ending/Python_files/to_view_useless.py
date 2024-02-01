import pandas as pd
from pandas.io.pytables import com

df = pd.read_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/1st_feb/dependent_control_variable_fully_filled_ADANIES_GDs_Below_100.csv",
    index_col=None,
)

comp_n = "MAXHEALT"


print(df.loc[df["Company Name"] == comp_n, "ALTMAN_Z_SCORE"])
