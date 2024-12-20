# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:31:33 2024

@author: julia
"""

# Import necessary libraries
import gurobipy as gp  
from gurobipy import *  
from gurobipy import GRB  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

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
demand_real = pd.read_excel(demand_data_path, skiprows=11, header=0)
demand_real = demand_real.set_index(demand_real.columns[1])
demand_real = demand_real.drop(demand_real.columns[0], axis=1)

#Create data for airport
airport_data = pd.read_excel(demand_data_path, skiprows=3, header=0)
airport_data = airport_data.drop(airport_data.columns[:2], axis=1)
airport_data = airport_data.drop(airport_data.index[5:])

# labels to rows
row_labels = ['ICOA codes', 'Latitude', 'Longitude', 'Runway', 'Slots']
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
n = len(Airports)  
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

#%%
#Part 1A      
# calculate lat and long to radials
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Initialize distance matrices
sigma = np.zeros((n, n))  # For haversine distances in radians
d = np.zeros((n, n))   

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

# Haversine Distance Calculation
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)
d = np.zeros((n, n))  # Distance matrix
for i in range(n):
    for j in range(n):
        delta_lat = lat_rad[i] - lat_rad[j]
        delta_lon = lon_rad[i] - lon_rad[j]
        a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[i]) * math.cos(lat_rad[j]) * math.sin(delta_lon / 2)**2
        sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d[i, j] = Re * sigma

# Filter non-zero demand and exclude self-demand
valid_indices = np.where((demand_real.values > 0) & (np.eye(n) == 0))
filtered_pop = np.log(pop_20.values.flatten()[valid_indices[0]] * pop_20.values.flatten()[valid_indices[1]] + epsilon)
filtered_gdp = np.log(gdp_20.values.flatten()[valid_indices[0]] * gdp_20.values.flatten()[valid_indices[1]] + epsilon)
filtered_d = np.log(f * d[valid_indices] + epsilon)
filtered_demand = np.log(demand_real.values[valid_indices] + epsilon)

# Regression
X = np.column_stack((filtered_pop, filtered_gdp, -filtered_d))
y = filtered_demand
model = LinearRegression()
model.fit(X, y)
b1, b2, b3 = model.coef_
c = model.intercept_
k = np.exp(c)

# Estimated Demand for 2020
pop_2020 = pop_20.values.flatten()
gdp_2020 = gdp_20.values.flatten()
estimated_demand_2020 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            estimated_demand_2020[i, j] = k * (
                (pop_2020[i] * pop_2020[j])**b1 * (gdp_2020[i] * gdp_2020[j])**b2
            ) / (f * d[i, j])**b3

# Flatten data for plotting
real_demand_2020 = demand_real.values.flatten()
estimated_demand_2020_flat = estimated_demand_2020.flatten()

# Scatter Plot: Real vs Estimated Demand
plt.figure(figsize=(10, 6))
plt.scatter(real_demand_2020, estimated_demand_2020_flat, alpha=0.7, label='Data Points')
plt.plot([real_demand_2020.min(), real_demand_2020.max()],
         [real_demand_2020.min(), real_demand_2020.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.title('Real Demand vs. Estimated Demand (2020)')
plt.xlabel('Real Demand 2020')
plt.ylabel('Estimated Demand 2020')
plt.legend()
plt.grid(True)
plt.show()

print(estimated_demand_2020)


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

# Plot Real vs. Estimated Demand for 2025
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


forecasted_demand_2025 = np.array(forecasted_demand_2025, dtype=int)
print(forecasted_demand_2025)

df = pd.DataFrame(forecasted_demand_2025)

# Save as Excel-file
excel_path = "forecasted_demand_2025.xlsx"
df.to_excel(excel_path, index=False)

#%%
#Part 1B
d = np.array(d)
q = np.array(forecasted_demand_2025)

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
f = 1.42                                   # Fuel cost per gallon
hub = 'EDDF'                               # Airport is data row 
R = [1500, 3300, 6300, 12000]              # range
RW_AC = [1400, 1600, 1800, 2600]           # Minimum runway length per aircraft at airport
TAT = [25/60, 35/60, 45/60, 60/60]         # Turnaround time (TAT) in minutes
M = 10000

g = []  # Binary variabel for hubs, 0 for hub, 1 for other airports
for i, airport in enumerate(Airports):
    if airport == hub:
        g.append(0)
    else:
        g.append(1)  # 

# Start modelling optimization problem
m = Model('practice')

# Decision Variables
# direct flow from airport i to airport j
x = {}             
for i in N:
    for j in N:
        x[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)

# number of flights from airport i to airport j
z = {}              
for i in N:
    for j in N:
        for k in K:
            z[i,j,k] = m.addVar(lb=0, vtype=GRB.INTEGER)

# flow from airport i to airport j that transfers at the hub
w = {}             
for i in N:
    for j in N:
        w[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)
        
# Number of aircrafts of type k
AC = {}             
for k in K:
    AC[k] = m.addVar(lb=0, vtype=GRB.INTEGER)
  
m.update()

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
                
                if g[i] == 0 or g[j] == 0:
                    cost = (Cx[k] + Ct + (Cf)) * 0.7 # Cost is only 70% when flight is to or from hub
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
con4 = {}
for i in N:
    for k in K:
        con4 = m.addConstr(quicksum(z[i, j, k] for j in N) == quicksum(z[j, i, k] for j in N), 
                name=f"FlowConservation_{i}")  # C3

# 5. Time constraint
con5 = {}    
for k in K:
    con5 = m.addConstr(
        quicksum(((d[i, j] / sp[k]) + TAT[k] * (1.5 - 0.5 * g[j])) * z[i, j, k]
            for i in N for j in N) <= BT * AC[k],
        name=f"TimeConstraint_{k}")
    
# 6. Aircraft range used to define matrix akij and constrain frequency to range limits
con6 = {}
for i in N:
    for j in N:
        for k in K:
            if d[i,j] <= R[k]:
                con6 = m.addConstr(z[i,j,k] <= M, name=f"range_constraint_{i}_{j}_{k}") # C5)
            else: 
                con6 = m.addConstr(z[i,j,k] <= 0, name=f"range_constraint_{i}_{j}_{k}") # C5


# 7. Runway airport must be bigger or equal to runway aircraft
con7 = {}
for i in N:
    for j in N:
        for k in K:
            if RW_AC [k] <= RW_AP[i] and RW_AC[k] <= RW_AP[j]:
                con7 = m.addConstr(z[i,j,k] <= M, name=f"runway_constraint_{i}_{j}_{k}")
            else:
                con7 = m.addConstr(z[i,j,k] <= 0, name=f"runway_constraint_{i}_{j}_{k}")


               
# Time limit
m.setParam('TimeLimit',590)

# Solve the model 
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

#%% Soluitons
#Print amout of aircraft per type
for k in K:
    print('Aircraft:', k,AC[k].X)

# Print Frequencies of flights
print("\nFrequencies:")
for i in N:
    for j in N:
        for k in K:
            if z[i, j, k].X > 0:
                print(f"{Airports[i]} to {Airports[j]}: {z[i, j, k].X}, with {Aircrafts[k]}")

#Print total amount of flights  
total_flights = 0
for i in N:
    for j in N:
        for k in K:
            total_flights += z[i, j, k].X
print("Total number of flights:", total_flights)

#Print total amount of flights involving the hub
sum_flights_to_hub = 0
for i in N:
    for j in N:
        for k in K:
            if g[i] == 0 or g[j] == 0:
                sum_flights_to_hub += z[i,j,k].X
print("Total number of flights involving EDDF:", sum_flights_to_hub)

# Printing flights per aircraft type
print("\nFlights per aircraft type:")
for k in K:
    print(f"\nAircraft Type {k} ({Aircrafts[k]}):")
    has_flights = False  # Flag to check if this aircraft type has flights
    for i in N:
        for j in N:
            if z[i, j, k].X > 0:
                print(f"  {Airports[i]} to {Airports[j]}: {z[i, j, k].X} flights")
                has_flights = True
    if not has_flights:
        print("  No flights for this aircraft type.")

# Calculate and print total hours per aircraft type
print("\nTotal hours per aircraft type:")
total_hours_per_aircraft = {}
for k in K:
    total_hours = 0  # Initialize total hours for aircraft type k
    for i in N:
        for j in N:
            if z[i, j, k].X > 0: 
                flight_hours = ((d[i, j]/ sp[k]) + (TAT[k]* (1.5 - 0.5 * g[j]))) * z[i, j, k].X 
                total_hours += flight_hours  
    total_hours_per_aircraft[k] = total_hours
    print(f"Aircraft Type {k} ({Aircrafts[k]}): {total_hours:.2f} hours")

# Print satisfied demand
satisfied_demand = 0
for i in N:
    for j in N:
        satisfied_demand += x[i, j].X + w[i, j].X  # Flow on direct and indirect flights

# Forecasted demand
forecasted_demand = np.sum(q)  # Total demand

# Satisfied demand
satisfied_demand_percentage = (satisfied_demand / forecasted_demand) * 100

# Print results
print(f"Total Satisfied Demand: {satisfied_demand}")
print(f"Total Forecasted Demand: {forecasted_demand}")
print(f"Satisfied Demand Percentage: {satisfied_demand_percentage:.2f}%")

#Print Yield
total_yield = 0
for i in N:
    for j in N:
        if x[i, j].X + w[i, j].X > 0:  # Alleen routes met passagiers
            route_yield = (5.9 * d[i, j] ** (-0.76) + 0.043) * d[i, j] * (x[i, j].X + w[i, j].X)
            total_yield += route_yield
print(f"Total Yield: {total_yield:.2f}")

# Calculate direct passengers
direct_passengers = 0
for i in N:
    for j in N:
        direct_passengers += x[i, j].X  

# Calculate hub-transfer passengers
hub_transfer_passengers = 0
for i in N:
    for j in N:
        hub_transfer_passengers += w[i, j].X

# Calculate total passengers
total_passengers = direct_passengers + hub_transfer_passengers

# Print results
print(f"Direct Passengers: {direct_passengers:.0f}")
print(f"Hub-Transfer Passengers: {hub_transfer_passengers:.0f}")
print(f"Total Passengers: {total_passengers:.0f}")

#Print flights on worldmap
def create_plot(freq, pax, lat=lat, long=lon):
    from shapely.geometry import Point
    import geopandas as gpd
    from geopandas import GeoDataFrame
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    
    # Plot airport locations
    gdf_reset.plot(ax=ax, marker='o', color='red', markersize=400, label="Airports")
    
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

                    # Adjust line width based on number of passengers
                    line_width = 0.5 + (total_num_pax / 10000)  # Scale factor for line thickness

                    # Plot the line (add label only once)
                    if not connections_plotted:
                        ax.plot([x_i, x_j], [y_i, y_j], linestyle='-', linewidth=line_width, color='blue', alpha=1, label="Connections")
                        connections_plotted = True
                    else:
                        ax.plot([x_i, x_j], [y_i, y_j], linestyle='-', linewidth=line_width, color='blue', alpha=1)
    
                    # Annotate the line with the total number of flights and pax
                    annotation_text = f'{total_num_pax}'
                    ax.annotate(annotation_text, (mid_x, mid_y), ha='center', va='center', fontsize=20, rotation=angle, color='darkred', weight='bold')

    # Add labels, title, and show plot
    plt.xlabel('Longitude in degrees', fontsize=25)
    plt.ylabel('Latitude in degrees', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("European Flight Network", fontsize=30)
    plt.legend(loc='lower left', fontsize=20)
    plt.grid(True)
    plt.show()



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


# List of airports
airports = Airports.tolist()

# Hub as central node
hub = "EDDF"  

# Routes 
edge_data = defaultdict(list)
for i in N:
    for j in N:
        for k in K:
            if z[i, j, k].X > 0:  # Only active routes
                source = airports[i]  # Source node
                target = airports[j]  # Target node
                edge_data[(source, target)].append(f"{Aircrafts[k]} - {z[i, j, k].X}")

# Make directed graph
G = nx.DiGraph()

# Show only active nodes
active_nodes = set()
for (source, target), data in edge_data.items():
    active_nodes.add(source)  # Source node
    active_nodes.add(target)  # Target node
    G.add_edge(source, target, label="\n".join(data))

G.add_nodes_from(active_nodes)

pos = nx.spring_layout(G, center=(0, 0), k=0.3, iterations=50)
pos = {node: pos[node] for node in active_nodes}  # Filter alleen actieve nodes

# Hub as central nodes
if hub in active_nodes:
    pos[hub] = (0, 0)

# Plot
plt.figure(figsize=(15, 15))
#Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
# Draw edges
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color="brown")
# Labels for nodes
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")
# Labels voor edges )
edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Plot 
plt.title("Flight Network with Aircraft Types and Number of Flights", fontsize=15)
plt.axis("off")
plt.show()


#%%
# Berekening van totale kosten en totale opbrengsten
total_costs = 0
total_revenue = 0

# Bereken totale kosten
for i in N:
    for j in N:
        for k in K:
            if z[i, j, k].X > 0:  # Alleen actieve routes
                # Tijdgebaseerde kosten
                Ct = ct[k] * (d[i, j] / sp[k])
                # Brandstofkosten
                Cf = ((cf[k] * f) / 1.50) * d[i, j]
                # Totale kosten per vlucht (inclusief korting via hub)
                if g[i] == 0 or g[j] == 0:  # Hub korting
                    cost = (Cx[k] + Ct + Cf) * 0.7
                else:
                    cost = Cx[k] + Ct + Cf
                total_costs += cost * z[i, j, k].X  # Kosten per vlucht * aantal vluchten

# Leasekosten per vliegtuigtype
for k in K:
    total_costs += AC[k].X * Cl[k]  # Leasekosten * aantal vliegtuigen van type k

# Bereken totale opbrengsten (revenue)
for i in N:
    for j in N:
        if x[i, j].X + w[i, j].X > 0:  # Routes met passagiers
            # Yield berekenen
            route_yield = (5.9 * d[i, j] ** (-0.76) + 0.043)
            # Totale opbrengst per route
            total_revenue += route_yield * (x[i, j].X + w[i, j].X)  * d[i, j]

# Resultaten printen
print(f"Total Costs: €{total_costs:.2f}")
print(f"Total Revenue: €{total_revenue:.2f}")

# Netto winst berekenen
net_profit = total_revenue - total_costs
print(f"Net Profit: €{net_profit:.2f}")

# Operating costs per aircraft type
operating_costs = {k: 0 for k in K}  # Initialiseer dictionary voor operating costs

# Bereken operating costs per vliegtuigtype
for k in K:
    for i in N:
        for j in N:
            if z[i, j, k].X > 0:  # Alleen actieve routes
                # Tijdgebaseerde kosten
                Ct = ct[k] * (d[i, j] / sp[k])
                # Brandstofkosten
                Cf = ((cf[k] * f) / 1.50) * d[i, j]
                # Totale kosten per vlucht
                if g[i] == 0 or g[j] == 0:  # Korting via hub
                    cost = (Cx[k] + Ct + Cf) * 0.7
                else:
                    cost = Cx[k] + Ct + Cf
                
                # Tel de kosten per vliegtuigtype op
                operating_costs[k] += cost * z[i, j, k].X

leasing_costs = {k: 0 for k in K}
# Leasekosten per vliegtuigtype toevoegen
for k in K:
    leasing_costs[k] += AC[k].X * Cl[k]

# Print operating costs per vliegtuigtype
print("\nOperating and lease Costs per Aircraft Type:")
for k in K:
    print(f"{Aircrafts[k]}: operating costs €{operating_costs[k]:,.2f}")
    print(f"{Aircrafts[k]}: leasing costs {leasing_costs[k]:,.2f}")
# Totale operating costs berekenen
tot_lease_costs= sum(leasing_costs.values())
total_operating_costs = sum(operating_costs.values())
print(f"\nTotal Operating Costs: €{total_operating_costs:,.2f}")
print(f"\nTotal Operating Costs: €{tot_lease_costs:,.2f}")
tot_costs = total_operating_costs + tot_lease_costs
print(f"\nTotal Costs: {tot_costs:,.2f}")

# Get the KPI

def calculate_kpis():
    # ASK 
    ask = 0
    for i in N:
        for j in N:
            for k in K:
                if z[i, j, k].X > 0:  # Alleen actieve routes
                    ask += s[k] * d[i, j] * z[i, j, k].X

    # RPK 
    rpk = 0
    for i in N:
        for j in N:
            if x[i, j].X + w[i, j].X > 0:  # Routes met passagiers
                rpk += (x[i, j].X + w[i, j].X) * d[i, j]

    # CASK 
    cask = total_costs / ask if ask > 0 else 0

    # RASK 
    rask = total_revenue / ask if ask > 0 else 0

    # Calculate Load Factor (LF) =
    ANload_factor = rpk / ask if ask > 0 else 0
    yield_value = (total_revenue - tot_costs + ask * cask) / rpk if rpk > 0 else 0
    # BELF 
    belf = cask / yield_value if yield_value > 0 else 0

    # Print results
    print(f"Available Seat Kilometers (ASK): {ask:,.2f}")
    print(f"Revenue Passenger Kilometers (RPK): {rpk:,.2f}")
    print(f"Cost per ASK (CASK): €{cask:.4f}")
    print(f"Revenue per ASK (RASK): €{rask:.4f}")
    print(f"Load Factor (ANLF): {ANload_factor:.2%}")
    print(f"Break-even Load Factor (BELF): {belf:.2%}")

calculate_kpis()
