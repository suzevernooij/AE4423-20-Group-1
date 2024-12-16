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
from scipy.spatial import distance_matrix
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
N = range(n)

lat = matrix[1].astype(float)  # Latitudes as float
lon = matrix[2].astype(float)  # Longitudes as float
RW_AP = matrix[3].astype(float).tolist() # range airports

# Print the extracted values
print(f"ICAO Codes (Nodes): {Airports}")
print(f"Latitudes: {lat}")
print(f"Longitudes: {lon}")

#Parameters
f = 1.42                #EUR/gallon #fuel costs
Re = 6371               #radius of the earth
        
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
for i in N:
    for j in N:
        # Compute the demand for the pair (i, j) using multiplication and division
        estimated_demand[i,j] = k * ((pop_20[i] * pop_20[j])**b1 * (gdp_20[i] * gdp_20[j])**b2) / (f * d[i, j])**b3


estimated_demand = np.array(estimated_demand, dtype=int)
print(estimated_demand)
print(demand_real)

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
forecasted_demand = np.array(forecasted_demand_2025, dtype=int)

# Print the forecasted demand matrix for 2025
print("Forecasted Demand for 2025:")
print(forecasted_demand)
print(estimated_demand)
print(demand_real)

# Totale vraag voor elke demand-matrix
total_forecasted_demand = np.sum(forecasted_demand)
total_estimated_demand = np.sum(estimated_demand)
total_real_demand = np.sum(demand_real.values)  # Gebruik .values als demand_real een pandas DataFrame is

# Print de resultaten
print("Totale vraag per demand-matrix:")
print(f"Forecasted Demand (2025): {total_forecasted_demand}")
print(f"Estimated Demand: {total_estimated_demand}")
print(f"Real Demand: {total_real_demand}")

#%%
d = np.array(d)
q = np.array(estimated_demand)
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
q = forecasted_demand                      # don't know if this works
f = 1.42                                   # Fuel cost per gallon
hub = 'EDDF'                               # Airport is data row 
R = [1500, 3300, 6300, 12000]              # range
RW_AC = [1400, 1600, 1800, 2600]           # Minimum runway length per aircraft at airport
TAT = [25/60, 35/60, 45/60, 60/60]  # Turnaround time (TAT) in minutes

g = []  # Binary variabel for hubs
for i, airport in enumerate(Airports):
    if airport == hub:
        g.append(0)
    else:
        g.append(1)  # 0 for hub, 1 for other airports
# y = {}
# for i in N:
#     for j in N:
#         if i != j:
#             y[i,j] = 5.9 * d[i,j]**(-0.76) + 0.043

# Ct={}
# for i in N:
#     for j in N:
#         if i != j:
#             for k in K:
#                 Ct[i, j, k] = ct[k] * (d[i,j] / sp[k])

# Cf={}
# for i in N:
#     for j in N:
#         if i != j:
#             for k in K:
#                 Cf[i,j,k] = ((cf[k] * f) / 1.5) * d[i,j]
                    
# C = {}
# for i in N:
#     for j in N:
#         if i != j:
#             for k in K:
#                 if g[i] == 0 or g[j] == 0:
#                     C[i,j,k] = (Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * 0.7 
#                 else:
#                     C[i,j,k] = Cx[k] + Ct[i,j,k] + Cf[i,j,k]  
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

AC = {}             # Number of aircrafts of type k
for k in K:
    AC[k] = m.addVar(lb=0, vtype=GRB.INTEGER)
  
m.update()


# m.setObjective(
#     quicksum(y[i,j] * d[i,j] * (x[i, j] + w[i,j]) for i in N for j in N if i != j) 
#     - quicksum(C[i,j,k] * z[i,j,k] for i in N for j in N for k in K if i != j)
#     - quicksum((Cl[k] * AC[k]) for k in K), GRB.MAXIMIZE
# )

obj = LinExpr()

###    YIELD
for i in N: 
    for j in N:
        if i!=j:
            YIELD = (5.9*d[i,j]**(-0.76))+0.043 
            obj += (YIELD * d[i,j] * (x[i,j] + w[i,j])) 
                
###    COSTS  
Costs = {}             
for i in N:
    for j in N:
        if i != j:
            for k in K:
                #Time based costs
                Ct = ct[k] * (d[i,j] / sp[k])
                
                #Fuel costs
                Cf = ((cf[k] * f) / 1.50) * d[i,j]
                
                #if Airports[j] == 'EDDF' or Airports[i] == 'EDDF':
                if g[i] == 0 or g[j] == 0:
                    cost = (Cx[k] + Ct + (Cf)) * 0.7 # Cost is reduced by 30% when flight is trough hub
                else:
                    cost = (Cx[k] + Ct + Cf)
                
                Costs[i,j,k] = cost
                obj -= Costs[i,j,k] * z[i,j,k]


for k in K:
    obj -= AC[k] * Cl[k]     
    
#Objective function                  
m.setObjective(obj , GRB.MAXIMIZE)
m.update()




# # Constraints
# 1. Flow should not exceed demand
con1 = {}
for i in N:
    for j in N:
        con1 = m.addConstr(x[i, j] + w[i,j] <= q[i,j], name=f"DemandConstraint_{i}_{j}")  # C1

# # 1*. Only consider transfer passangers if the hub is not origin or destination  
con2 = {}      
for i in N:
    for j in N:
        con2 = m.addConstr(w[i,j] <= q[i,j] * g[i] * g[j], name=f"TransferPassangers_{i}_{j}") #C1*


# # # 2. Flow is limited by capacity
con3 = {}
for i in N:
    for j in N:
        con3 = m.addConstr(
            x[i, j] + quicksum(w[i, m] * (1 - g[j]) for m in N) + quicksum(w[m, j] * (1 - g[i]) for m in N) 
            <= quicksum(z[i, j, k] * s[k] * LF for k in K), 
            name=f"CapacityConstraint_{i}_{j}")

# 3. Flow conservation: flow in equals flow out for each airport
con5 = {}
for i in N:
    for k in K:
        con5 = m.addConstr(quicksum(z[i, j, k] for j in N) == quicksum(z[j, i, k] for j in N), 
                name=f"FlowConservation_{i}")  # C3

# # con6 = {}
# # for k in K:
# #     con6 = m.addConstr(
# #         quicksum(((d[i, j] / sp[k]) + TAT[k] * (1.5 if g[j] == 0 else 1) * z[i, j, k])
# #             for i in N for j in N) <= BT * AC[k],
# #         name=f"TimeConstraint_{k}")
        
# # con6 = {}
# # for k in K:
# #     if g[j] == 0:
# #         con6 = m.addConstr(
# #             quicksum(((d[i, j] / sp[k]) + TAT[k] * (1.5) * z[i, j, k])
# #                 for i in N for j in N) <= BT * AC[k],
# #             name=f"TimeConstraint_{k}")
# #     else:
# #         con6 = m.addConstr(
# #             quicksum(((d[i, j] / sp[k]) + TAT[k] * z[i, j, k])
# #                 for i in N for j in N) <= BT * AC[k],
# #             name=f"TimeConstraint_{k}")
# # for k in K:
# #     if Airports[j] == 'EDDF':
# #         con6 = m.addConstr((((d[i, j] / sp[k]) + TAT[k] * (1.5) * z[i, j, k])
# #             for i in N for j in N) <= BT * AC[k])
# #     else:
# #         con6 = m.addConstr((((d[i, j] / sp[k]) + TAT[k] * (1) * z[i, j, k])
# #             for i in N for j in N) <= BT * AC[k])
con6 = {}    
for k in K:
    con6 = m.addConstr(
        quicksum(((d[i, j] / sp[k]) + TAT[k] * (1.5 - 0.5 * g[j])) * z[i, j, k]
            for i in N for j in N) <= BT * AC[k],
        name=f"TimeConstraint_{k}")
    
# 5. Aircraft range used to define matrix akij and constrain frequency to range limits
con7 = {}
for i in N:
    for j in N:
        for k in K:
            if d[i,j] <= R[k]:
                con7 = m.addConstr(z[i,j,k] <= 100000, name=f"range_constraint_{i}_{j}_{k}") # C5)
            else: 
                con7 = m.addConstr(z[i,j,k] <= 0, name=f"range_constraint_{i}_{j}_{k}") # C5


# 6. Runway airport must be bigger or equal to runway aircraft
con8 = {}
for i in N:
    for j in N:
        for k in K:
            if RW_AC [k] <= RW_AP[i] and RW_AC[k] <= RW_AP[j]:
                con8 = m.addConstr(z[i,j,k] <= 100000, name=f"runway_constraint_{i}_{j}_{k}")
            else:
                con8 = m.addConstr(z[i,j,k] <= 0, name=f"runway_constraint_{i}_{j}_{k}")


                    
# Solve the model
# m.optimize()

# # Check the status of the optimization
# status = m.status


# if status == GRB.Status.UNBOUNDED:
#     print('The model cannot be solved because it is unbounded')
# elif status == GRB.Status.OPTIMAL:
#     f_objective = m.objVal
#     print('***** RESULTS ******')
#     print('\nObjective Function Value: \t %g' % f_objective)
# elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
#     print('Optimization was stopped with status %d' % status)

# # Print out Solutions for frequencies (z[i,j])
# print("\nFrequencies:----------------------------------")
# for i in N:
#     for j in N:
#         if z[i, j, k].X > 0:
#             print(f"{Airports[i]} to {Airports[j]}: {z[i, j, k].X}")

# Stel de tijdslimiet in
m.setParam('TimeLimit',700)

# Los het model op
m.optimize()
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
        for k in K:
            if z[i, j, k].X > 0:
                print(f"{Airports[i]} to {Airports[j]}: {z[i, j, k].X}")
                
#%%
# Printing total amount of aircraft
for k in K:
    print('Aircraft:', k,AC[k].X)
print("\nFrequencies:----------------------------------")
for i in N:
    for j in N:
        for k in K:
            if z[i, j, k].X > 0:
                print(f"{Airports[i]} to {Airports[j]}: {z[i, j, k].X}")

#Printing total amount of flights  
total_flights = 0
for i in N:
    for j in N:
        for k in K:
            total_flights += z[i, j, k].X
print("Total number of flights:", total_flights)

#Printing total amount of flights involving "LIME"
sum_flights_to_hub = 0

for i in N:
    for j in N:
        for k in K:
            if g[i] == 0 or g[j] == 0:
                sum_flights_to_hub += z[i,j,k].X
print("Total number of flights involving EDDF:", sum_flights_to_hub)


def create_plot(freq, pax, lat=lat, long=lon):
    from shapely.geometry import Point
    import geopandas as gpd
    from geopandas import GeoDataFrame
    import matplotlib.pyplot as plt
    
    # Load the world map shapefile
    world = gpd.read_file(r'C:\Users\julia\Downloads\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp')
    
    # Filter for European countries
    europe = world[world['CONTINENT'] == 'Europe']
    
    # Add Madeira manually as a single point
    madeira = GeoDataFrame(geometry=[Point(-16.9595, 32.7607)], crs='EPSG:4326')
    
    # Create a GeoDataFrame for airport locations
    geometry = [Point(xy) for xy in zip(long, lat)]
    gdf_reset = GeoDataFrame(geometry=geometry, crs='EPSG:4326')
    
    # Plot setup
    fig_size_a0 = (46.811, 33.1102)
    fig, ax = plt.subplots(figsize=fig_size_a0)
    
    # Plot Europe and Madeira
    europe.plot(ax=ax, color='lightgray', edgecolor='black', label='European Countries')
    madeira.plot(ax=ax, marker='o', color='blue', markersize=500, label="Madeira")
    
    # Plot airport locations
    gdf_reset.plot(ax=ax, marker='o', color='red', markersize=500, label="Airports")
    
    # Set the boundaries for Europe and Madeira
    ax.set_xlim([-25, 45])  # Longitude (includes Madeira)
    ax.set_ylim([30, 75])   # Latitude
    
    # Print airport ICAO codes
    for x, y, label in zip(gdf_reset.geometry.x + 0.5, gdf_reset.geometry.y + 0.5, airport_data.loc['ICOA codes']):
        if ax.get_xlim()[0] <= x <= ax.get_xlim()[1] and ax.get_ylim()[0] <= y <= ax.get_ylim()[1]:
            ax.text(x, y, label, fontsize=24)
            
    # Initialize a flag to ensure "Connections" appears only once in the legend
    connections_plotted = False
    
    # Plot lines with frequency and number of passengers
    for i in N:
        for j in N:
            if i < j:
                num_flights_ij = freq.iloc[i, j]
                num_flights_ji = freq.iloc[j, i]
                total_num_flights = num_flights_ij + num_flights_ji
    
                num_pax_ij = pax.iloc[i, j]
                num_pax_ji = pax.iloc[j, i]
                total_num_pax = num_pax_ij + num_pax_ji
    
                # Check if the total number of flights is greater than zero
                if total_num_flights > 0 and total_num_pax > 0:
                    x_i, y_i = gdf_reset.loc[i, 'geometry'].x, gdf_reset.loc[i, 'geometry'].y
                    x_j, y_j = gdf_reset.loc[j, 'geometry'].x, gdf_reset.loc[j, 'geometry'].y
    
                    # Calculate the midpoint for the line and place the text
                    mid_x = (x_i + x_j) / 2
                    mid_y = (y_i + y_j) / 2
                    
                    # Calculate the angle of the line
                    angle = np.degrees(np.arctan2(y_j - y_i, x_j - x_i))
                    
                    # Ensure the text is not upside down by adjusting the angle
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
    
                    # Plot the line (add label only once)
                    if not connections_plotted:
                        ax.plot([x_i, x_j], [y_i, y_j], linestyle='-', linewidth=0.5, color='blue', label="Connections")
                        connections_plotted = True
                    else:
                        ax.plot([x_i, x_j], [y_i, y_j], linestyle='-', linewidth=0.5, color='blue')
    
                    # Annotate the line with the total number of flights and pax
                    annotation_text = f'{total_num_pax}({total_num_flights})'
                    ax.annotate(annotation_text, (mid_x, mid_y + 1), ha='center', va='center', fontsize=16, rotation=angle)

    # Add labels, title, and show plot
    plt.xlabel('Longitude in degrees', fontsize=25)
    plt.ylabel('Latitude in degrees', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("European Flight Network Including Madeira", fontsize=30)
    plt.legend(loc='lower left', fontsize=20)
    plt.grid(True)
    plt.show()



def freq_pax_to_excel(freq_array, pax_array, airport=airport):
    combined_df = pd.DataFrame(index=airport.iloc[:, 0], columns=airport.iloc[:, 0])
    freq_df = pd.DataFrame(index=airport.iloc[:, 0], columns=airport.iloc[:, 0])
    pax_df = pd.DataFrame(index=airport.iloc[:, 0], columns=airport.iloc[:, 0])

    # Iterate through the cells of freq_array and pax_array to populate the new DataFrames
    for i, icao_i in enumerate(airport.iloc[:, 0]):
        for j, icao_j in enumerate(airport.iloc[:, 0]):
            pax = int(pax_array.iloc[i, j])
            flights = int(freq_array.iloc[i, j])

            # Combine the information in the desired format
            combined_info = f'{pax}({flights})' if pax > 0 else '-'

            # Assign the combined information to the corresponding cell in the combined DataFrame
            combined_df.at[icao_i, icao_j] = combined_info

            # Assign individual values to freq_df and pax_df
            freq_df.at[icao_i, icao_j] = flights
            pax_df.at[i, j] = pax

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full file path for the Excel file in the same directory as the script
    excel_path = os.path.join(script_directory, 'combined_data.xlsx')

    # Save all DataFrames to an Excel file
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        combined_df.to_excel(writer, sheet_name='Combined', index=True)
        freq_df.to_excel(writer, sheet_name='Frequency', index=True)
        pax_df.to_excel(writer, sheet_name='Passengers', index=True)

    # Print a message indicating the successful save
    return f'The combined data, frequency, and passengers have been saved to {excel_path}'

total_flights_array = {}
total_pax = {}
total_transfer = {}
for i in N:
    total_flights_array[i] = {}
    total_pax[i] = {}
    total_transfer[i] = {}
    for j in N:
        total_flights_array[i][j] = sum(z[i, j, k].X for k in K)  # Sum of flights
        total_pax[i][j] = w[i, j].X + x[i, j].X  # Total passengers
        total_transfer[i][j] = w[i, j].X  # Transfer passengers

total_flights_df = pd.DataFrame(total_flights_array)  # Flights DataFrame
total_pax_df = pd.DataFrame(total_pax)               # Passengers DataFrame
total_transfer_df = pd.DataFrame(total_transfer)     # Transfer DataFrame



print(create_plot(freq=total_flights_df, pax=total_pax_df))
