import pandas as pd

df = pd.read_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/1st_feb/dependent_control_variable_fully_filled_ADANIES_GDs_Below_100.csv",
    index_col=None,
)


df_1 = df[df["Year"] > 2008]

df_1.to_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/1st_feb/dependent_control_variable_fully_filled_ADANIES_GDs_Below_100_lst_15_years.csv"
)
