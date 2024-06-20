import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/FOr_service_and_energy/energy/"

econ_df_1 = pd.read_csv(
    os.path.join(file_path, "time_lag.csv"),
    index_col=0,
)

econ_df_1 = econ_df_1[econ_df_1["Year"] > 2007]
econ_df_1 = pd.DataFrame(econ_df_1)
# set the index to the year column
econ_df_1 = econ_df_1.set_index("Year")
for column_name in econ_df_1.columns:
    if econ_df_1[column_name].isnull().values.any():
        print(column_name)
        index = econ_df_1[column_name].index[econ_df_1[column_name].apply(np.isnan)]
        # inds = pd.isnull(econ_df_1).any(column_name).nonzero()[0]
        df_index = econ_df_1.index.values.tolist()
        inds = [df_index.index(i) for i in index]
        print(inds)


econ_df = econ_df_1.drop(["Company Name"], axis=1)
# set the data type and select rows up to 2016
econ_df = econ_df.astype(float)
econ_df_after = econ_df.drop(
    ["ENVIRON_DISCLOSURE_SCORE", "SOCIAL_DISCLOSURE_SCORE", "GOVNCE_DISCLOSURE_SCORE"],
    axis=1,
)

dependent_variables = ["RETURN_ON_ASSET", "RETURN_COM_EQY"]

for values_i in dependent_variables:
    print(values_i)
    Y = econ_df_after[[values_i]]
    X = econ_df_after.drop(["RETURN_ON_ASSET", "RETURN_COM_EQY"], axis=1)
    df = X.copy()

    # Add constant term to the independent variable
    X = sm.add_constant(X)

    # Fit robust GLS model
    model = sm.GLS(Y, X)
    results = model.fit(cov_type="HC3")
    # print(results.summary())

    name = values_i + "_GLS.txt"

    # Save the model summary to a text file
    with open(os.path.join(file_path, "GLS_results", name), "w") as file:
        file.write(str(results.summary()))
    #     #
    #     # Fit robust GLS model
#     # results = model.fit(cov_type="robust")
print("Done")
