import os
from matplotlib import axis
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm

# from statsmodels.stats.stattools import wald_test
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

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
econ_df_after = econ_df.drop(
    ["ENVIRON_DISCLOSURE_SCORE", "SOCIAL_DISCLOSURE_SCORE", "GOVNCE_DISCLOSURE_SCORE"],
    axis=1,
)

dependent_variables = ["RETURN_ON_ASSET", "TOBIN_Q_RATIO", "ALTMAN_Z_SCORE"]

for values_i in dependent_variables:
    print(values_i)
    Y = econ_df_after[[values_i]]
    X = econ_df_after.drop(
        ["ALTMAN_Z_SCORE", "RETURN_ON_ASSET", "TOBIN_Q_RATIO"], axis=1
    )
    # Split X and y into X_
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=1
    )

    # create a Linear Regression model object
    regression_model = LinearRegression()

    # pass through the X_train & y_train data set
    regression_model.fit(X_train, y_train)

    # let's grab the coefficient of our model and the intercept
    intercept = regression_model.intercept_[0]
    coefficent = regression_model.coef_[0][0]

    print("The intercept for our model is {:.4}".format(intercept))
    print("-" * 50)
    # loop through the dictionary and print the data
    for coef in zip(X.columns, regression_model.coef_[0]):
        print("The Coefficient for {} is {:.2}".format(coef[0], coef[1]))

    # define our intput
    X2 = sm.add_constant(X)

    # create a OLS model
    model = sm.OLS(Y, X2)

    # fit the data
    est = model.fit()
    # Run the White's test
    # Heteroscedasticity test (Breusch-Pagan test)
    _, pval, _, f_pval = sm.stats.diagnostic.het_breuschpagan(est.resid, est.model.exog)
    print(pval, f_pval)
    print("-" * 50)

    # print the results of the test
    if pval > 0.05:
        print("For the Breusch-Pagan's Test")
        print("The p-value was {:.4}".format(pval))
        print(
            "We fail to reject the null hypthoesis, so there is no heterosecdasticity."
        )

    else:
        print("For the Breusch-Pagan's Test")
        print("The p-value was {:.4}".format(pval))
        print("We reject the null hypthoesis, so there is heterosecdasticity.")

    # Get fitted values and residuals
    fitted_values = est.fittedvalues
    residuals = est.resid

    # Plot residuals vs. fitted values

    titla_value = "Residuals vs. Fitted Values of " + values_i
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at y=0
    plt.title(titla_value)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()
    print("#" * 100)

    titla_value = "Residuals vs. log(Fitted Values of " + values_i
    plt.scatter(np.log(fitted_values), residuals)
    plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at y=0
    plt.title(titla_value)
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.show()
    print("#" * 100)
exit()
model1 = sm.OLS(Y, X2)

# fit the data
est = model.fit()
# Run the White's test
# Heteroscedasticity test (Breusch-Pagan test)
_, pval, _, f_pval = sm.stats.diagnostic.het_breuschpagan(est.resid, est.model.exog)
print(pval, f_pval)
print("-" * 100)

# print the results of the test
if pval > 0.05:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

else:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity.")

# Get fitted values and residuals
fitted_values = est.fittedvalues
residuals = est.resid

# Plot residuals vs. fitted values
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at y=0
plt.title("Residuals vs. Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()
exit()
# print(pval, f_pval)
# print("-" * 100)
#
# # print the results of the test
# if pval > 0.05:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print(
#         "We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n"
#     )
#
# else:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity. \n")
#
# # Run the Breusch-Pagan test
# _, pval, __, f_pval = diag.het_breuschpagan(est.resid, est.model.exog)
# print(pval, f_pval)
# print("-" * 100)
#
# # print the results of the test
# if pval > 0.05:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")
#
# else:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity.")
