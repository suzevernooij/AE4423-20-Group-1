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
pop_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\pop.xlsx'
demand_data_path = r'C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 1\Assignment 1A\DemandGroup1.xlsx'

# #import data Julia
# pop_data_path = r'C:\TIL\Jaar 2\Airline Planning\pop.xlsx'
# demand_data_path = r'C:\TIL\Jaar 2\Airline Planning\DemandGroup1.xlsx'

#Create data for population
pop = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=0)
pop = pop.drop(pop.columns[2:], axis=1)

#Create data for GDP
gdp = pd.read_excel(pop_data_path, skiprows=2, header=0, index_col=4)
gdp = gdp.drop(gdp.columns[:4], axis=1)

#Create data for demand
demand = pd.read_excel(demand_data_path, skiprows=11, header=0)
demand = demand.drop(demand.columns[0], axis=1)

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
print(f"ICAO Codes (Nodes): {N}")
print(f"Latitudes: {latitudes}")
print(f"Longitudes: {longitudes}")

#Parameters
f = 1.42                #EUR/gallon #fuel costs
Re = 6371               #radius of the earth

# Create array for euclidian distances between nodes (1 distance unit corresponds with 1 time unit)
sigma = np.zeros((n,n))    
for i in range(n):
    for j in range(n):
        sigma[i,j] = 2*math.asin(math.sqrt(math.sin((lat[i]-lat[j])/2)**2+math.cos(lat[i])*math.cos(lat[j])*math.sin((lon[i]-lon[j])/2)**2)) 

d = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        d[i,j]= Re * sigma[i,j]
        
print(d[0,1])

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

print(matrix_pop)

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

print(matrix_gdp)



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