from operator import le
import os
from matplotlib import axis
import numpy as np
import pandas as pd
from pyparsing import col
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# from statsmodels.stats.stattools import wald_test
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt,
    het_arch,
    acorr_ljungbox,
)

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
# residuals_1 = pd.Series([])
exog_1 = np.empty((1, 7))
for values_i in dependent_variables:
    print(values_i)
    Y = econ_df_after[[values_i]]
    X = econ_df_after.drop(["RETURN_ON_ASSET", "RETURN_COM_EQY"], axis=1)
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
    model = sm.OLS(Y, X2).fit()

    # fit the data
    # est = model.fit()
    # Run the White's test
    # print(est.model.exog)
    #
    # # Perform the White test
    # Get the residuals
    # Get the exogenous variables for the White test
    residuals = model.resid
    # Perform Goldfeld-Quandt test
    gq_test_statistic, gq_p_value, split_point = het_goldfeldquandt(model.resid, X)
    print(f"Goldfeld-Quandt Test Statistic: {gq_test_statistic}")
    print(f"P-value: {gq_p_value}")

    if gq_p_value >= 0.05:
        print("For the Goldfeld-Quandt Test")
        print("The p-value was {:.4}".format(gq_p_value))
        print(
            "We fail to reject the null hypthoesis, so there is no heterosecdasticity."
        )
        print("We accept null hypothesis, so there is homocedacity")

    else:
        print("For the Goldfeld-Quandt Test")
        print("The p-value was {:.4}".format(gq_p_value))
        print("We reject the null hypthoesis, so there is heterosecdasticity.")

    arch_test = het_arch(model.resid)
    print("ArCH test")
    print(arch_test)
    # Get the residuals from the fitted model
    resid = model.resid

    # cd_test = cdtest(model, group_effect=True)
    # Perform the Arellano-Bond test
    # Get the residuals from the fitted model
    resid = model.resid

    # Perform the Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(
        resid, lags=10
    )  # Specify the number of lags (n_lags) to test

    # Print the test results
    print("Ljung-Box Test Results:")
    print(lb_test)
    # Perform the Pesaran CD test
    # pesaran_cd_test = cd(cd.Pesaran, model, w, test_type="cd")
    # print(pesaran_cd_test.summary)
    # # # plt.figure(figsize=(8, 6))
    # # # plt.scatter(fitted_values, residuals, alpha=0.8)
    # # # plt.xlabel("Fitted values")
    # # # plt.ylabel("Residuals")
    # # # if values_i == "RETURN_ON_ASSET":
    # # #     plt.title("Residuals vs Fitted Values of Return on Asset")
    # # # else:
    # # #     plt.title("Residuals vs Fitted Values of Return on Common Equity")
    # # # plt.axhline(y=0, color="k", linestyle="--")  # Add horizontal line at y=0
    # # # # Set limits for x-axis and y-axis
    # # # plt.ylim(bottom=-30, top=30)  # Example limits for the y-axis
    # # # plt.show()
    # # # # residuals_1 = residuals_1.append(residuals, ignore_index=True)
    # # #
    # Assuming 'model' is your fitted regression model
    bg_test_stat, bg_p_value, _, _ = acorr_breusch_godfrey(model)
    print("Breusch_godfrey p-value: ", bg_p_value)
    # Check for autocorrelation
    if bg_p_value < 0.05:
        print("Significant autocorrelation detected.")
    else:
        print("No significant autocorrelation detected.")
    # # #
    # # # plt.rcParams["figure.figsize"] = (18, 10)
    # # # titla_value = "Distribution of Error Term"
    # # # plt.hist(
    # # #     residuals, bins="auto", density=True, alpha=0.7, color="blue", edgecolor="black"
    # # # )
    # # # # Fit a normal distribution to the data
    # # # mu, std = norm.fit(residuals)
    # # #
    # # # # Plot the PDF of the fitted distribution
    # # # xmin, xmax = plt.xlim()
    # # # x = np.linspace(xmin, xmax, 100)
    # # # p = np.exp(-((x - mu) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
    # # # plt.plot(
    # # #     x,
    # # #     p,
    # # #     "k",
    # # #     linewidth=2,
    # # #     label="Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (mu, std),
    # # # )
    # # # plt.title(titla_value)
    # # # plt.xlabel("Residuals")
    # # # plt.ylabel("Frequency")
    # # # plt.grid(True)
    # # # plt.show()
    # # # # plt.rcParams["figure.figsize"] = (19, 10)
    # # # # plt.rcParams["figure.figsize"] = (18, 10)
    # # # # titla_value = "Distribution of Error Term"
    # # # # plt.hist(
    # # # #     residuals, bins="auto", density=True, alpha=0.7, color="blue", edgecolor="black"
    # # # # )
    # # # # # Fit a normal distribution to the data
    # # # # mu, std = norm.fit(residuals_1)
    # # # #
    # # # # # Plot the PDF of the fitted distribution
    # # # # xmin, xmax = plt.xlim()
    # # # # x = np.linspace(xmin, xmax, 100)
    # # # # p = np.exp(-((x - mu) ** 2) / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
    # # # # plt.plot(
    # # # #     x,
    # # # #     p,
    # # # #     "k",
    # # # #     linewidth=2,
    # # # #     label="Fit results: $\mu$ = %.2f, $\sigma$ = %.2f" % (mu, std),
    # # # # )
    # # # # plt.title(titla_value)
    # # # # plt.xlabel("Residuals")
    # # # # plt.ylabel("Frequency")
    # # # # plt.grid(True)
    # # # # plt.show()