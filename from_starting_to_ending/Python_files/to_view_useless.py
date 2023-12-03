import pickle
import pandas as pd

file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/independent_variables_wasted_companies_dict.pkl"

# open a file, where you stored the pickled data
file = open(file_path, "rb")

# dump information to that file
wasted_companies = pickle.load(file)


# close the file
file.close()

df = pd.read_csv(
    "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/dependent_control_variable_filled.csv"
)

df_copy = df.copy()
i = 0
for val in wasted_companies:
    if len(wasted_companies[val]) != 0:
        i += 1
        print(val)
        print(f"{i}) {val} : {wasted_companies[val]}")
