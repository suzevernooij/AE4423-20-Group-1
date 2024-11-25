# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:25:42 2024

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

#name model
m = Model('AirlinePlanningAssingment1')

#import data Suze
# pop_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\pop.xlsx'
# demand_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\DemandGroup1.xlsx'

# #import data Julia
pop_data_path = r'C:\TIL\Jaar 2\Airline Planning\pop.xlsx'
demand_data_path = r'C:\TIL\Jaar 2\Airline Planning\DemandGroup1.xlsx'

#Create data for population
pop = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=0)
pop = pop.drop(pop.columns[2:], axis=1)
pop_20 = pop.drop(pop.columns[1:], axis=1)
pop_2 = pop.drop(pop.columns[2:], axis=1)
pop_23 = pop.drop(pop_2.columns[:1], axis=1)

#Create data for GDP
gdp = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=4)
gdp = gdp.drop(gdp.columns[:4], axis=1)
gdp_20 = gdp.drop(gdp.columns[1:], axis=1)
gdp_2 = gdp.drop(gdp.columns[2:], axis=1)
gdp_23 = gdp.drop(gdp_2.columns[:1], axis=1)

#Create data for demand
demand_real = pd.read_excel(demand_data_path, skiprows=12, header=0)
demand_real = demand.drop(demand.columns[0], axis=1)

#Create data for airport
airport_data = pd.read_excel(demand_data_path, skiprows=4, header=0)
airport_data = airport_data.drop(airport_data.columns[:2], axis=1)
airport_data = airport_data.drop(airport_data.index[4:])

row_labels = ['Latitude', 'Longitude', 'Runway', 'Slots']

# Instellen van de index met labels
airport_data.index = row_labels

#Print data 
print(pop.head())
print(gdp.head())
print(demand.head())
print(airport_data.tail())

# N = airport_data.index[0]
# n = len(N)
# print(N)
# print(n)

# Create array for data related to nodes
matrix = []  # Create array to store data
i = 0  # Counter for processing data
for line in airport_data.values:
    row = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            row.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            row.append(item)
    matrix.append(row)

matrix = np.array(matrix)

# Extract Nodes (ICAO codes), Latitudes, and Longitudes
N = matrix[0]  # ICAO Codes as nodes
n = len(N)  # Number of airports/nodes

lat = matrix[0].astype(float)  # Latitudes as float
lon = matrix[1].astype(float)  # Longitudes as float

# Print the extracted values
# print(f"ICAO Codes (Nodes): {N}")
# print(f"Latitudes: {latitudes}")
# print(f"Longitudes: {longitudes}")

#Parameters
f = 1.42                #EUR/gallon #fuel costs
Re = 6371               #radius of the earth

# Create array for euclidian distances between nodes (1 distance unit corresponds with 1 time unit)
sigma = np.zeros((n,n))    
for i in range(n):
    for j in range(n):
            sigma[i,j] = 2*math.asin(math.sqrt(math.sin((lat[i]-lat[j])/2)**2+math.cos(lat[i])*math.cos(lat[j])*math.sin((lon[i]-lon[j])/2)**2)) 
1
d = np.zeros((n,n))
for i in range(n):
    for j in range(n):
            d[i,j]= Re * sigma[i,j]
        
print(d)

matrix_pop = []
i = 0  # Counter for processing data
for line in pop.values:
    column = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            column.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            column.append(item)
    matrix_pop.append(column)

matrix_pop = np.array(matrix_pop)

matrix_gdp = []
i = 0  # Counter for processing data
for line in gdp.values:
    column = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            column.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            column.append(item)
    matrix_gdp.append(column)

matrix_gdp = np.array(matrix_gdp)

matrix_demand = []
i = 0  # Counter for processing data
for line in demand_real.values:
    column = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            column.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            column.append(item)
    matrix_demand.append(column)

matrix_demand = np.array(matrix_demand)


matrix_pop_20 = []
i = 0  # Counter for processing data
for line in pop_20.values:
    column = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            column.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            column.append(item)
    matrix_pop_20.append(column)

matrix_pop_20 = np.array(matrix_pop_20)

print(matrix_pop_20)

matrix_gdp_20 = []
i = 0  # Counter for processing data
for line in gdp_20.values:
    column = []
    for item in line:
        # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
        try:
            # Try to convert to float, as latitudes/longitudes are floats
            column.append(float(item))
        except ValueError:
            # If the conversion fails (non-numeric values like ICAO code), append as string
            column.append(item)
    matrix_gdp_20.append(column)

matrix_gdp_20 = np.array(matrix_gdp_20)

# demand = []
# for i in range(n):
#     for j in range(n):
#         demand[i,j] = math.log(k) + b1 * math.log(pop_20[i]*pop_20[j]) + b2 * math.log(gdp_20[i]*gdp_20[j] - b3 * math.log(f*d[i,j]))
        
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assuming `observed_demand` is a numpy array or pandas DataFrame of observed demand values.
# Replace `observed_demand` with your actual dataset.

# log_demand = np.log(demand_real)  # Dependent variable (Y)

# # Avoid log(0) or negative values
# epsilon = 1e-6

# # Compute X1, X2, X3 with epsilon to avoid invalid log operations
# X1 = np.log(np.outer(pop_20.flatten(), pop_20.flatten()) + epsilon)
# X2 = np.log(np.outer(gdp_20.flatten(), gdp_20.flatten()) + epsilon)
# X3 = np.log(f * d + epsilon)

# # Dependent variable
# log_demand = np.log(observed_demand + epsilon)

# # # Prepare the independent variables (X1, X2, X3)
# # X1 = np.log(np.outer(pop_20, pop_20))  # Outer product of populations
# # X2 = np.log(np.outer(gdp_20, gdp_20))  # Outer product of GDPs
# # X3 = np.log(f * d_safe)  # Distance-related term

# # Flatten the arrays into 1D for regression
# X = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten()))
# Y = log_demand.flatten()

# # Fit the regression model
# reg = LinearRegression()
# reg.fit(X, Y)

# # Extract the coefficients
# a = reg.intercept_  # This is ln(k)
# b1, b2, b3 = reg.coef_  # These are the coefficients

# # Convert ln(k) back to k
# k = np.exp(a)

# # Print the results
# print(f"k: {k}")
# print(f"b1: {b1}")
# print(f"b2: {b2}")
# print(f"b3: {b3}")


# Avoid log(0) or negative values
epsilon = 1e-6

# Compute X1, X2, X3 with epsilon to avoid invalid log operations
X1 = np.log(np.outer(pop_20, pop_20) + epsilon)
X2 = np.log(np.outer(gdp_20, gdp_20) + epsilon)
X3 = np.log(f * d + epsilon)

# Dependent variable
log_demand = np.log(demand_real + epsilon)

# Combine X1, X2, X3 into a feature matrix
X = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten()))
# Y = log_demand.flatten()

# Remove invalid rows
valid_mask = ~(
    np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) |
    np.isnan(Y) | np.isinf(Y)
)
X_filtered = X[valid_mask]
Y_filtered = Y[valid_mask]

# Fit regression model
reg = LinearRegression()
reg.fit(X_filtered, Y_filtered)

# Extract coefficients
a = reg.intercept_  # ln(k)
b1, b2, b3 = reg.coef_

# Convert ln(k) back to k
k = np.exp(a)

# Print results
print(f"k: {k}")
print(f"b1: {b1}")
print(f"b2: {b2}")
print(f"b3: {b3}")

import matplotlib.pyplot as plt
import numpy as np

# Step 1: Calculate estimated demand
estimated_log_demand = (
    a + 
    b1 * np.log(np.outer(pop_20, pop_20) + epsilon) +
    b2 * np.log(np.outer(gdp_20, gdp_20) + epsilon) -
    b3 * np.log(f * d + epsilon)
)
estimated_demand = np.exp(estimated_log_demand)  # Transform back to original scale

# Step 2: Flatten real demand for comparison
real_demand = demand_real.values

# Step 3: Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(real_demand, estimated_demand, alpha=0.7, label='Data Points')
plt.plot([real_demand.min(), real_demand.max()], 
         [real_demand.min(), real_demand.max()], 
         color='red', linestyle='--', label='Perfect Fit')  # Line of equality

plt.ylim(0, 1000)
# Step 4: Add labels and legend
plt.title('Real Demand vs. Estimated Demand (2020)')
plt.xlabel('Real Demand')
plt.ylabel('Estimated Demand')
plt.legend()
plt.grid(True)
plt.show()



# matrix_pop = []  # Initialize as a Python list, not a numpy array
# i = 0  # Counter for processing data
# for line in pop.values:
#     column = []
#     for item in line:
#         # Only convert numeric values (like Latitude/Longitude/Runway/Slots) to float or int
#         try:
#             # Try to convert to float, as latitudes/longitudes are floats
#             column.append(float(item))
#         except ValueError:
#             # If the conversion fails (non-numeric values like ICAO code), append as string
#             column.append(item)
#     matrix_pop.append(column)  # Append to the list, not to a numpy array
# matrix = np.array(matrix)
# pop_20 = matr
# matrix = []                                    # Create array for data related to nodes
# i=0                                            # to keep track of lines in data file
# for line in airport_data:
#     i=i+1
#     words = line.split()
#     words=[int(i) for i in words]           # Covert data from string to integer
#     matrix.append(words)                       # Store node data
# matrix = np.array(matrix)

# N           =matrix[:,0]                       # Nodes
# n=len(N) 
# print(n)


# #Parameters
# f = 1.42                #EUR/gallon #fuel costs
# Re = 6371               #radius of the earth

# # lat = matrix.index[1]
# # lon = matrix.index[2]

# lat = airport_data.iloc[:, 1].values  # Extract latitudes (assuming latitudes are in the 2nd column)
# lon = airport_data.iloc[:, 2].values  # Extract longitudes (assuming longitudes are in the 3rd column)


# # Create array for euclidian distances between nodes (1 distance unit corresponds with 1 time unit)
# sigma = np.zeros((n,n))    
# for i in range(n):
#     for j in range(n):
#         sigma[i,j] = 2*math.asin(math.sqrt(math.sin((lat[i]-lat[j])/2)**2+math.cos(lat[i])*math.cos(lat[j])*math.sin((lon[i]-lon[j])/2)**2)) 
        

# for i in range(n):
#     for j in range(n):
#         # Haversine formula to calculate distance between (lat[i], lon[i]) and (lat[j], lon[j])
#         dlat = math.radians(lat[i] - lat[j])  # Convert to radians
#         dlon = math.radians(lon[i] - lon[j])  # Convert to radians
#         a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat[i])) * math.cos(math.radians(lat[j])) * math.sin(dlon / 2) ** 2
#         c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#         sigma[i, j] = Re * c  # Distance in kilometers