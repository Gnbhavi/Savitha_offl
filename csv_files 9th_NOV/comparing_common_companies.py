import os

import pandas as pd


def getting_the_value(path, dropping_comp_list):
    """This is for finding the coreect values"""
    print("The list of companies to find the values")
    scores = pd.read_csv(path)
    scores = pd.DataFrame(scores)
    scores = scores["0"].to_list()
    company_name = []
    for companies_1 in scores:
        if companies_1 not in dropping_comp_list:
            company_name.append(companies_1)
    print(company_name)

    return company_name


base_path = "~/Python_github/savitha/Savitha_offl/"
dropping_comp = pd.read_excel(
    os.path.join(base_path, "csv_files 9th_NOV/Drop_list_90_firms.xlsx")
)
dropping_comp = pd.DataFrame(dropping_comp)
dropping_comp.columns = dropping_comp.iloc[0]
dropping_comp = dropping_comp.drop(dropping_comp.index[0])

dropping_comp_list = dropping_comp["Housing Finance Company"]
dropping_comp_list = dropping_comp_list[19:].tolist()
dropping_company_name = []
for companies in dropping_comp_list:
    dropping_company_name.append(companies.replace(" IN Equity", ""))

the_column_names_to_check = [
    "Altman_missing_list.csv",
    "BS_CUR_LIAB.csv",
    "BS_CUR_ASSET_REPORT.csv",
    "Non_cur_liab_missing_companies.csv",
]

companies_with_scores = {}
for column_values in the_column_names_to_check:
    print(column_values[:-4])
    companies_with_scores[column_values] = getting_the_value(
        os.path.join(base_path, "Jupyter files", column_values), dropping_company_name
    )
exit()
altman_missing_companies = getting_the_value(
    "~/Python_github/savitha/Savitha_offl/Jupyter files/Altman_missing_list.csv"
)
# Current Liability
bs_cur_liab_missing_comp = getting_the_value(
    "~/Python_github/savitha/Savitha_offl/Jupyter files/BS_CUR_LIAB.csv"
)

# Current Asset
bs_cur_asset_missing_comp = getting_the_value(
    "~/Python_github/savitha/Savitha_offl/Jupyter files/BS_CUR_ASSET_REPORT.csv"
)
# Non Current liabilities
bs_non_cur_liab_missing_comp = getting_the_value(
    "~/Python_github/savitha/Savitha_offl/Jupyter files/Non_cur_liab_missing_companies.csv"
)

altman_unwanted_companies = []
current_liab_unwanted_comp = []
current_asset_unwanted_comp = []
non_current_liab_unwanted_comp = []
for companies in dropping_comp_list:
    companyname = companies.replace(" IN Equity", "")

    if companyname in altman_missing_companies:
        altman_unwanted_companies.append(companyname)

    if companyname in bs_cur_liab_missing_comp:
        current_liab_unwanted_comp.append(companyname)

    if companyname in bs_cur_asset_missing_comp:
        current_asset_unwanted_comp.append(companyname)

    if companyname in bs_non_cur_liab_missing_comp:
        non_current_liab_unwanted_comp.append(companyname)


print("Number of companies that wanted Altman score: ", len(altman_missing_companies))
print("The number of altman unwanted companies: ", len(altman_unwanted_companies))
print(altman_unwanted_companies)

print(
    "Number of companies that wanted current liabilities: ",
    len(bs_cur_liab_missing_comp),
)
print(
    "The number of current liabilities unwanted companies: ",
    len(current_liab_unwanted_comp),
)
print(current_liab_unwanted_comp)

print("Number of companies that wanted Current Asset: ", len(bs_cur_asset_missing_comp))
print(
    "The number of Current asset unwanted companies: ", len(current_asset_unwanted_comp)
)
print(current_asset_unwanted_comp)

print(
    "Number of companies that wanted Non Current Liability: ",
    len(bs_non_cur_liab_missing_comp),
)
print(
    "The number of non-current liabilities unwanted companies: ",
    len(non_current_liab_unwanted_comp),
)
print(non_current_liab_unwanted_comp)
