import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import probplot
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_breuschpagan


file_path = "/Users/gnbhavithran/Python_github/savitha/Savitha_offl/from_starting_to_ending/new_files/"
econ_df_1 = pd.read_csv(os.path.join(file_path, "timeLag_2003.csv"), index_col=0)
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
independent_variables = [
    "ENVIRON_DISCLOSURE_SCORE",
    "SOCIAL_DISCLOSURE_SCORE",
    "GOVNCE_DISCLOSURE_SCORE",
    "ESG_DISCLOSURE_SCORE",
]
dependent_variables = ["RETURN_ON_ASSET", "TOBIN_Q_RATIO", "ALTMAN_Z_SCORE"]

for ind_variables in independent_variables:
    drop_vals = [x for x in independent_variables if x != ind_variables]
    econ_df_after = econ_df.drop(drop_vals, axis=1)
    for values_i in dependent_variables:
        print(drop_vals, " and ", values_i)
        y = econ_df_after[[values_i]]
        X = econ_df_after.drop(
            ["ALTMAN_Z_SCORE", "RETURN_ON_ASSET", "TOBIN_Q_RATIO"], axis=1
        )
        # Fit GLS regression model
        X_with_constant = sm.add_constant(X)  # Add a constant term
        model = sm.GLS(y, X_with_constant)
        results = model.fit()

        # Display regression results
        print(results.summary())
        name = (
            values_i + " and " + ind_variables.replace("_DISCLOSURE_SCORE", "") + ".txt"
        )
        # Save the model summary to a text file
        with open(os.path.join(file_path, name), "w") as file:
            file.write(str(results.summary))
        # Assuming 'results' is your GLS regression results
        residuals = results.resid
        # Assuming 'results' is your GLS regression results
        # probplot(residuals, plot=plt)
        # plt.title("Q-Q Plot of Residuals")
        # plt.show()
        bp_test = het_breuschpagan(results.resid, results.model.exog)
        print("Breusch-Pagan test p-value:", bp_test[1])
        # Assuming 'results' is your GLS regression results
        # lb_test = acorr_ljungbox(results.resid)
        # print("Ljung-Box test p-values:", lb_test[1])
        # sns.histplot(residuals, kde=True)
        # plt.title("Histogram of Residuals")
        # plt.show()
        print("R-squared:", results.rsquared)
        print("Adjusted R-squared:", results.rsquared_adj)
        print("-" * 50)
    print("*" * 75)
