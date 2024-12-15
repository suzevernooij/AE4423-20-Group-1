# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:07:57 2024

@author: suzev
"""

# Import necessary libraries
import gurobipy as gp  # Gurobi optimization library
from gurobipy import *  # Import all Gurobi functions
from gurobipy import GRB  # Import Gurobi constants
import numpy as np  # For numerical operations
import math  # For mathematical functions
import copy  # For copying objects
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.linear_model import LinearRegression

# Name the model
m = Model('AirlinePlanningAssignment1')

#import data Suze
pop_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\pop.xlsx'
demand_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\DemandGroup1.xlsx'

# Create data for population
pop = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=0)
pop = pop.drop(pop.columns[2:], axis=1)
pop_20 = pop.drop(pop.columns[1:], axis=1)
pop_2 = pop.drop(pop.columns[2:], axis=1)
pop_23 = pop.drop(pop_2.columns[:1], axis=1)

# Create data for GDP
gdp = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=4)
gdp = gdp.drop(gdp.columns[:4], axis=1)
gdp_20 = gdp.drop(gdp.columns[1:], axis=1)
gdp_2 = gdp.drop(gdp.columns[2:], axis=1)
gdp_23 = gdp.drop(gdp_2.columns[:1], axis=1)

# Create data for demand
demand_real = pd.read_excel(demand_data_path, skiprows=11, header=0)
demand_real = demand_real.set_index(demand_real.columns[1])
demand_real = demand_real.drop(demand_real.columns[0], axis=1)

# Create data for airport
airport_data = pd.read_excel(demand_data_path, skiprows=3, header=0)
airport_data = airport_data.drop(airport_data.columns[:2], axis=1)
airport_data = airport_data.drop(airport_data.index[5:])

row_labels = ['ICAO codes', 'Latitude', 'Longitude', 'Runway', 'Slots']
airport_data.index = row_labels

# Extract Nodes (ICAO codes), Latitudes, and Longitudes
matrix = np.array(airport_data.values)
Airports = matrix[0]  # ICAO Codes as nodes
n = len(Airports)  # Number of airports/nodes

lat = matrix[1].astype(float)  # Latitudes as float
lon = matrix[2].astype(float)  # Longitudes as float
RW_AP = matrix[3].astype(float) # Runway availability

# Parameters
f = 1.42                # EUR/gallon # fuel costs
Re = 6371               # Radius of the earth

# Calculate lat and long to radians
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Initialize distance matrices
sigma = np.zeros((n, n))  # For haversine distances in radians
d = np.zeros((n, n))      # For actual distances in kilometers

# Calculate the Haversine distances between all pairs of cities
for i in range(n):
    for j in range(n):
        # Differences in latitude and longitude
        delta_lat = lat_rad[i] - lat_rad[j]
        delta_lon = lon_rad[i] - lon_rad[j]
        
        # Haversine formula
        a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[i]) * math.cos(lat_rad[j]) * math.sin(delta_lon / 2)**2
        sigma[i, j] = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Distances in kilometers
        d[i, j] = Re * sigma[i, j]

# Step 1: Filter non-zero demand and exclude self-demand
valid_indices = np.where((demand_real.values > 0) & (np.eye(n) == 0))

# Filter data for regression
epsilon = 1e-11  # Small value to avoid log(0)
filtered_pop = np.log(pop_20.values.flatten()[valid_indices[0]] * pop_20.values.flatten()[valid_indices[1]] + epsilon)
filtered_gdp = np.log(gdp_20.values.flatten()[valid_indices[0]] * gdp_20.values.flatten()[valid_indices[1]] + epsilon)
filtered_d = np.log(f * d[valid_indices] + epsilon)
filtered_demand = np.log(demand_real.values[valid_indices] + epsilon)

# Combine into regression matrix
X = np.column_stack((filtered_pop, filtered_gdp, -filtered_d))
y = filtered_demand

# Step 2: Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Extract parameters
b1, b2, b3 = model.coef_
c = model.intercept_
k = np.exp(c)

# Print the coefficients
print(f"b1: {b1}")
print(f"b2: {b2}")
print(f"b3: {b3}")
print(f"k: {k}")

# Forecast population and GDP for 2025
years = 2025 - 2023
growth_rate_pop = (pop_23.values / pop_20.values) ** (1 / 3) - 1  
pop_25 = pop_23.values * (1 + growth_rate_pop)**years

growth_rate_gdp = (gdp_23.values / gdp_20.values) ** (1 / 3) - 1  
gdp_25 = gdp_23.values * (1 + growth_rate_gdp)**years

# Ensure no zero population or GDP
def replace_zeros_with_k(matrix, k):
    return np.where(matrix == 0, k, matrix)

pop_25 = replace_zeros_with_k(pop_25, k)
gdp_25 = replace_zeros_with_k(gdp_25, k)

# Forecast demand for 2025 using the gravity model
forecasted_demand_2025 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:  # Exclude self-demand
            forecasted_demand_2025[i, j] = k * (
                (pop_25[i] * pop_25[j])**b1 * (gdp_25[i] * gdp_25[j])**b2
            ) / (f * d[i, j])**b3

# Print the forecasted demand matrix for 2025
print("Corrected Forecasted Demand for 2025:")
print(forecasted_demand_2025)

# Step 3: Plot Real vs. Estimated Demand for 2025
real_demand = demand_real.values.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(real_demand, forecasted_demand_2025.flatten(), alpha=0.7, label='Data Points')
plt.plot([real_demand.min(), real_demand.max()], 
         [real_demand.min(), real_demand.max()], 
         color='red', linestyle='--', label='Perfect Fit')
plt.title('Real Demand vs. Estimated Demand (2025)')
plt.xlabel('Real Demand')
plt.ylabel('Estimated Demand')
plt.legend()
plt.grid(True)
plt.show()

print(forecasted_demand_2025)