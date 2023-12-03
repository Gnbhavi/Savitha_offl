import pandas as pd
import numpy as np

df = pd.read_csv(
    "~/Python_github/savitha/Savitha_offl/Nov_19_23/env_dis_scr_filled.csv"
)

df = pd.DataFrame(df)

null_positions = np.where(df.isnull())
null_positions_1 = list(zip(null_positions[0], null_positions[1]))
print(null_positions_1)
