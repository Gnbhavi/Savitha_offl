import os
import time
import math
import numpy as np
import pandas as pd

# Read the excel file
def excel_file_reader(excel_file, sheet_names):
    df = {}
    for sheet_name in sheet_names:
        df[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
    return df


def pivoit_table(list_table_to_change, column_count):
    transposed_list_table = list(map(list, zip(*list_table_to_change)))
    from itertools import chain

    matrix2 = []
    for k1 in range(column_count):
        values = []
        for i in range(k1, len(transposed_list_table), 14):
            values.append(transposed_list_table[i])
        flattened_list = list(chain.from_iterable(values))
        matrix2.append(flattened_list)
    final_table = list(map(list, zip(*matrix2)))
    return final_table


def company_name_creator(company_list, year_count):
    company_list_new =[]
    for values_hell in company_list:
        for i in range(year_count):
            company_list_new.append(values_hell)
    print(len(company_list_new))
    return company_list_new

# Read the first excel file
savitha_1_path = "02_09_23/savita.xlsx"         # Put the path of the excel file
sheet_names = ['Sheet3','Sheet4']               # Put the sheet names of the excel file
company_sheet = 'Sheet3'                        # Put the sheet name of the company
savitha1 = excel_file_reader(savitha_1_path, sheet_names)       # Read the excel file
remove_word = " IN Equity"                      
DAtes = savitha1['Sheet4'].iloc[3:, 0]          # Put the column name of the dates
Total_companies_count = 1086                    # Put the total number of companies
Dates = pd.to_datetime(DAtes).dt.date.astype(str).tolist()
Dates = Dates * Total_companies_count
Dates = pd.DataFrame(Dates, columns=['Dates'])
companies = [company[0].replace(remove_word, "") for company in savitha1[company_sheet].values]
                                       #Remove the  above line if you want to dont want toremove the word

number_of_headings = 14                                          # Put the number of headings in the excel file
columns_1 = savitha1['Sheet4'].iloc[2, 1:15].values.tolist()     # Put the column names of the excel file
selected_data = savitha1['Sheet4'].iloc[3:, 1:]                  # Put the data of the excel file
savitha1_paneled_list = selected_data.values.tolist()
savitha1_paneled_list = pivoit_table(savitha1_paneled_list, number_of_headings)
savitha1_paneled_list = pd.DataFrame(savitha1_paneled_list, columns=columns_1)