import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CorrelationsWithPath:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)   # For Reading CSV file

    def get_correlations(self, columns, storing_location):
        corr = self.df[columns].corr()   # For fiding correlation between columns
        corr.to_csv(storing_location)    # Stores the correlation in the given location
        return corr
