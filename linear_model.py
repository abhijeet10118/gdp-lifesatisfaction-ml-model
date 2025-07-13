import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

# The missing function definition (standard from the original source)
def prepare_country_stats(oecd_bli, gdp_per_capita):
    # This function merges the two datasets and prepares them for the linear regression.
    
    # 1. Prepare OECD BLI data
    # Select 'Country', 'Indicator', 'Value', and filter by 'Life satisfaction'
    oecd_bli = oecd_bli[oecd_bli["Indicator"]=="Life satisfaction"]
    # Pivot the table to have Country as index and Indicator/Unit as columns (flatten the data)
    oecd_bli = oecd_bli.pivot_table(columns='Indicator', index='Country', values='Value')
    
    # 2. Prepare GDP per capita data
    # Rename columns for clarity and set 'Country' as index
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    # 3. Merge the datasets
    full_country_stats = pd.merge(oecd_bli, gdp_per_capita, 
                                  left_index=True, right_index=True)
    
    # 4. Sort and select relevant columns (optional, but good practice for consistency)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)

    # 5. Return the prepared dataframe
    return full_country_stats

# Note: You also need to ensure pandas is imported (which you already have at the top of your script).


# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
 encoding='latin1', na_values="n/a")
# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()
# Select a linear model
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# Train the model
model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]
