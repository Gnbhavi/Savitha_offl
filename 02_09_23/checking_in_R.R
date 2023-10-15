# Load necessary libraries
library(zoo)

# Sample data (replace this with your actual data)
years <- c(2000, 2001, 2002, 2003, 2006, 2007, 2008, 2009, 2010)
revenue <- c(NA, NA, NA, NA, 100, 120, 140, 160, 200)

# Create a zoo object for the data
data_zoo <- zoo(revenue, years)

# Perform linear interpolation to fill missing values
data_filled <- na.approx(data_zoo)

# Print the filled data
print(data_filled)
