import pandas as pd
import statsmodels.api as sm
from linearmodels import PanelOLS

class MultipleRegression:
    def __init__(self, data):
        self.data = data
        self.data["year"] = pd.to_datetime(self.data["year"], format="%Y")
        self.data = self.data.set_index(["entity", "year"])
        # self.dependent_variable = dependent_variable
        # self.independent_variables = independent_variables

    def  model_regression(self, dependent_variable, independent_variables):
        panel_data_model = PanelOLS(dependent_variable, independent_variables, entity_effects=True)
        return panel_data_model.fit()
    
if __name__ == "__main__":
    data_set_path = input("Enter the path of the data set: ")
    dependent_variable = input("Enter the dependent variable: ")
    independent_variables = input("Enter the independent variables: ")
    data = pd.read_csv(data_set_path, index_col=0)
    panel_data = pd.DataFrame(data)
    MR = MultipleRegression(panel_data)
    results = {}
    for values in dependent_variable:
        panel_data_model = PanelOLS(panel_data[values], panel_data[independent_variables], entity_effects=True)
        results[values] = panel_data_model.fit()