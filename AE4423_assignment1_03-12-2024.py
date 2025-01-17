# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:31:33 2024

@author: julia
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
# demand = pd.read_excel(demand_data_path, skiprows=11, header=0)
# demand_real = demand.drop(demand.columns[0], axis=1)
demand_real = pd.read_excel(demand_data_path, skiprows=11, header=0)
demand_real = demand_real.set_index(demand_real.columns[1])
demand_real = demand_real.drop(demand_real.columns[0], axis=1)

#Create data for airport
airport_data = pd.read_excel(demand_data_path, skiprows=3, header=0)
airport_data = airport_data.drop(airport_data.columns[:2], axis=1)
airport_data = airport_data.drop(airport_data.index[5:])

row_labels = ['ICOA codes', 'Latitude', 'Longitude', 'Runway', 'Slots']

# Instellen van de index met labels
airport_data.index = row_labels

# #Print data 
# print(pop.head())
# print(gdp.head())
# print(demand.head())
# print(airport_data.tail())

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
Airports = matrix[0]  # ICAO Codes as nodes
n = len(Airports)  # Number of airports/nodes

lat = matrix[1].astype(float)  # Latitudes as float
lon = matrix[2].astype(float)  # Longitudes as float
RW_AP = matrix[3].astype(float) # range airports

# Print the extracted values
print(f"ICAO Codes (Nodes): {Airports}")
print(f"Latitudes: {lat}")
print(f"Longitudes: {lon}")

#Parameters
f = 1.42                #EUR/gallon #fuel costs
Re = 6371               #radius of the earth

# # Create array for euclidian distances between nodes (1 distance unit corresponds with 1 time unit)
# sigma = np.zeros((n,n))    
# for i in range(n):
#     for j in range(n):
#         sigma[i,j] = 2*math.asin(math.sqrt(math.sin((lat[i]-lat[j])/2)**2+math.cos(lat[i])*math.cos(lat[j])*math.sin((lon[i]-lon[j])/2)**2)) 

# d = np.zeros((n,n))
# for i in range(n):
#     for j in range(n):
#         d[i,j]= Re * sigma[i,j]
          
# print(d)

# calculate lat and long to radials
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Initialiseer afstandsmatrices
sigma = np.zeros((n, n))  # Voor haversine-afstanden in radialen
d = np.zeros((n, n))      # Voor daadwerkelijke afstanden in kilometers

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

epsilon = 1e-11  # A small value to replace zero distances

X1 = np.log(np.outer(pop_20, pop_20) + epsilon)
X2 = np.log(np.outer(gdp_20, gdp_20)+ epsilon)
X3 = -np.log(1.42 * d + epsilon)  # Fuel cost * distances
y = np.log(demand_real.values.flatten()+ epsilon)


# Perform linear regression
X = np.column_stack((X1.flatten(), X2.flatten(), X3.flatten()))
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

pop_20 = np.where(pop_20 == 0, k, pop_20)
gdp_20 = np.where(gdp_20 == 0, k, gdp_20)

# Loop over each pair (i, j)
estimated_demand = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        # Compute the demand for the pair (i, j) using multiplication and division
        estimated_demand[i,j] = k * ((pop_20[i] * pop_20[j])**b1 * (gdp_20[i] * gdp_20[j])**b2) / (f * d[i, j])**b3

print(estimated_demand)

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

# Ordinary Least Square Method
import numpy as np
# Add a column of ones to X for the intercept term
X = np.column_stack((np.ones(X.shape[0]), X))  # Adding a column of 1's for the intercept

# Compute the OLS estimate of coefficients (beta)
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Predicted values of y
y_pred = X @ beta_hat

# Residuals (difference between actual and predicted values)
residuals = y - y_pred

# Sum of squared residuals (RSS)
rss = np.sum(residuals**2)

# Print the estimated coefficients (beta) and other results
print(f"Estimated Coefficients: {beta_hat}")
print(f"Residuals: {residuals}")
print(f"Sum of Squared Residuals (RSS): {rss}")

## Forecast population and GDP for 2025
years = 2025 - 2023
growth_rate_pop = (pop_23 / pop_20) ** (1 / 3) - 1  
pop_25 = pop_23 * (1 + growth_rate_pop)**years

growth_rate_gdp = (gdp_23 / gdp_20) ** (1 / 3) - 1  
gdp_25 = gdp_23 * (1 + growth_rate_gdp)**years

print(pop_25)
print(gdp_25)

## Future demand 2025
# Forecast demand for 2025 using the gravity model

pop_25 = np.where(pop_25 == 0, k, pop_25)
gdp_25 = np.where(gdp_25 == 0, k, gdp_25)

forecasted_demand_2025 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        # Use the same gravity model formula with forecasted population and GDP for 2025
        forecasted_demand_2025[i, j] = k * ((pop_25[i] * pop_25[j])**b1 * (gdp_25[i] * gdp_25[j])**b2) / (f * d[i, j])**b3

# Print the forecasted demand matrix for 2025
print("Forecasted Demand for 2025:")
print(forecasted_demand_2025)

# # Visualize Real Demand (2020) vs. Forecasted Demand (2025)
# plt.figure(figsize=(10, 6))
# plt.scatter(real_demand.flatten(), forecasted_demand_2025.flatten(), alpha=0.7, label='Data Points')
# plt.plot([real_demand.min(), real_demand.max()], 
#          [real_demand.min(), real_demand.max()], 
#          color='red', linestyle='--', label='Perfect Fit')

# plt.title('Real Demand (2020) vs. Forecasted Demand (2025)')
# plt.xlabel('Real Demand (2020)')
# plt.ylabel('Forecasted Demand (2025)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Data
N = range(len(d))
Aircrafts = ['AC1', 'AC2', 'AC3','AC4']    # Aircraft types
K = range(len(Aircrafts))                  # Aircraft types number
Cx = [300, 600, 1250, 2000]                # Fixed operating cost per flight leg per aircraft type
ct = [750, 775, 1400, 2800]                # Flight cost per hour per aircraft type
cf = [1, 2, 3.75, 9]                       # Fuel cost parameter per aircraft type 
Cl = [15000, 34000, 80000, 190000]         # Weekly lease cost per aircraft type
LF = 0.75                                  # Minimum load factor per flight leg
s = [45, 70, 150, 320]                     # Number of seats per aircraft 
sp = [550, 820, 850, 870]                  # Speed of aircraft type k
BT = 10 * 7                                # Maximum operating time per aircraft 
y = 0.18                                   # yield still needs to be calculated with the formula, appendice A
q = estimated_demand                       # don't know if this works
f = 1.42                                   # Fuel cost per gallon
hub = 'EDDF'                               # Airport is data row 
R = [1500, 3300, 6300, 12000]              # range
RW_AC = [1400, 1600, 1800, 2600]           # Minimum runway length per aircraft at airport
# RW_AP = airport_data[7]                  # connect airport data runway
TAT_nohub = [25, 35, 45, 60]  # Turnaround time (TAT) in minutes
TAT_nohub_hours = [tat / 60 for tat in TAT_nohub]  # Convert TAT_nohub to hours

TAT_hub = [tat * 1.5 / 60 for tat in TAT_nohub]  # Convert TAT_hub to hours

y = {}
for i in N:
    for j in N:
        y[i,j] = 5.9 * d[i,j]**(-0.76) + 0.043

Ct={}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                Ct[i, j, k] = ct[k] * d[i,j] / sp[k]

Cf={}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                Cf[i,j,k] = ((cf[k] * f) / 1.5) * d[i][j]
                    
# C = {}
# for i in N:
#     for j in N:
#         if i != j:
#             for k in K:
#                 if Airports[i] == hub or Airports[j] == hub:
#                     C[i,j,k] = (Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * 0.7 
#                 else:
#                     C[i,j,k] = Cx[k] + Ct[i,j,k] + Cf[i,j,k] 
                    
# Lease costs need to be taken into account

g = {}  # Binary variabel for hubs
for i, airport in enumerate(Airports):
    g[i] = 0 if airport == hub else 1  # 0 for hub, 1 for other airports
    
# Start modelling optimization problem
m = Model('practice')

# Decision Variables
x = {}              # direct flow from airport i to airport j
for i in N:
    for j in N:
        x[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)

z = {}              # number of flights from airport i to airport j
for i in N:
    for j in N:
        for k in K:
            z[i,j,k] = m.addVar(lb=0, vtype=GRB.INTEGER)
        
w = {}              # flow from airport i to airport j that transfers at the hub
for i in N:
    for j in N:
        w[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)

AC = {}
for k in K:
    AC[k] = m.addVar(lb=0, vtype=GRB.INTEGER)
  
m.update()

m.setObjective(
    quicksum(y[i,j] * d[i,j] * (x[i, j] + w[i,j]) for i in N for j in N if i != j) 
    - quicksum((Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * z[i, j, k] + (Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * w[i, j] * 0.7 for i in N for j in N for k in K if i != j)
    - quicksum((Cl[k] * AC[k]) for k in K), GRB.MAXIMIZE
)

# Constraints
# 1. Flow should not exceed demand
for i in N:
    for j in N:
        m.addConstr(x[i, j] + w[i,j] <= q[i][j], name=f"DemandConstraint_{i}_{j}")  # C1

# 1*. Only consider transfer passangers if the hub is not origin or destination        
for i in N:
    for j in N:
        m.addConstr(w[i,j] <= q[i][j] * g[i] * g[j], name=f"TransferPassangers_{i}_{j}") #C1*

# # 2. Flow is limited by capacity
# # for i in N:
# #     for j in N:
# #         m.addConstr(x[i, j] <= quicksum(z[i, j, k] * s[k] * LF for k in K), name=f"CapacityConstraint_{i}_{j}")  # C2
        
# Constraint: x_ij + sum(w_im * (1 - g_j)) + sum(w_mj * (1 - g_i)) <= z_ij * s * LF
for i in N:
    for j in N:
        m.addConstr(
            x[i, j] + quicksum(w[i, m] * (1 - g[j]) for m in N) + quicksum(w[m, j] * (1 - g[i]) for m in N) 
            <= quicksum(z[i, j, k] * s[k] * LF for k in K), 
            name=f"CapacityConstraint_{i}_{j}")

# 3. Flow conservation: flow in equals flow out for each airport
for i in N:
    for k in K:
        m.addConstr(quicksum(z[i, j, k] for j in N) == quicksum(z[j, i, k] for j in N), 
                name=f"FlowConservation_{i}")  # C3

# # 4. Total time constraint for the flights 
# for k in K:
#     if j == hub:
#         m.addConstr(quicksum(quicksum((d[i,j] / sp[k] + (TAT_hub[k]/60)) * z[i, j, k] for i in N) for j in N) <= BT * AC[k], 
#                 name="TimeConstraint")  # C4
#     else:
#         m.addConstr(quicksum(quicksum((d[i,j] / sp[k] + (TAT_nohub[k]/60)) * z[i, j, k] for i in N) for j in N) <= BT * AC[k], 
#                 name="TimeConstraint")  # C4

 
# for k in K:
#     m.addConstr(
#         quicksum((d[i][j] / sp[k] + (TAT_hub[k])) * z[i, j, k] for i in N for j in N if j == hub) <= BT * AC[k],
#         name=f"TotalTimeConstraint_k{k}")

# for k in K:
#     m.addConstr(
#         quicksum((d[i][j] / sp[k] + (TAT_hub[k])) * z[i, j, k] for i in N for j in N if j == hub) +
#         quicksum((d[i][j] / sp[k] + (TAT_nohub_hours[k])) * z[i, j, k] for i in N for j in N if j != hub) <= BT * AC[k],
#         name=f"TotalTimeConstraint_k{k}")

# Add the constraint for each aircraft k
for k in K:
    m.addConstr(
        quicksum(
            (d[i][j] / sp[k] + TAT[j, k] * (1.5 if j == hub else 1)) * z[i, j, k]
            for i in N for j in N if i != j  # Ensure i != j to avoid diagonal (same destination)
        ) <= BT * AC[k],  # Total time constraint for each aircraft k
        name=f"TotalTimeConstraint_k{k}"
    )

    
# 5. Aircraft range used to define matrix akij and constrain frequency to range limits
ak = {}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                if d[i,j] <= R[k]:
                    m.addConstr(z[i,j,k] <= 100000, name=f"range_constraint_{i}_{j}_{k}") # C5)
                else: 
                    m.addConstr(z[i,j,k] <= 0, name=f"range_constraint_{i}_{j}_{k}") # C5

# 6. Runway airport must be bigger or equal to runway aircraft
for i in N:
    for j in N:
        if i != j:
            for k in K:
                if RW_AC [k] <= RW_AP[i]:
                    m.addConstr(z[i,j,k] <= 10000, name=f"runway_constraint_{i}_{j}_{k}")
                else:
                    m.addConstr(z[i,j,k] <= 0, name=f"runway_constraint_{i}_{j}_{k}")
                    
                    
# Solve the model
m.optimize()

# Check the status of the optimization
status = m.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
elif status == GRB.Status.OPTIMAL:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)
elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

# Print out Solutions for frequencies (z[i,j])
print("\nFrequencies:----------------------------------")
for i in N:
    for j in N:
        if z[i, j, k].X > 0:
            print(f"{Airports[i]} to {Airports[j]}: {z[i, j, k].X}")
